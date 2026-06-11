//! Differential tests for the wasm→IR transpiler: assemble WAT, transpile to our IR, **verify** it,
//! then run on both the interpreter and the JIT and check the result against a hand-computed oracle.
//! Verifying proves the transpiler emits well-formed, escape-safe IR; interp==JIT is the usual oracle.

use svm_interp::Value;

/// Transpile WAT → IR, verify, then run the export `entry` with `args` on interp + JIT; assert both
/// return the same single i64 and return it.
fn run(wat: &str, entry: &str, args: &[Value]) -> i64 {
    let wasm = wat::parse_str(wat).expect("assemble wat");
    let t = svm_wasm::transpile(&wasm).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify transpiled IR");
    let idx = t
        .exports
        .iter()
        .find(|(n, _)| n == entry)
        .unwrap_or_else(|| panic!("no export {entry}"))
        .1;

    let results = &t.module.funcs[idx as usize].results;
    let mut fuel = 100_000_000u64;
    let interp = svm_interp::run(&t.module, idx, args, &mut fuel).expect("interp run");
    let jit_args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(x) => *x as i64,
            Value::I64(x) => *x,
            other => panic!("unsupported arg {other:?}"),
        })
        .collect();
    let jit = match svm_jit::compile_and_run(&t.module, idx, &jit_args).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    // Compare per result type, normalizing i32 to its 32-bit pattern (the interp carries a typed i32;
    // the JIT a raw i64 whose high bits are ABI-defined) — sign/zero-extension isn't a transpiler concern.
    assert_eq!(jit.len(), interp.len(), "result count");
    for (i, rt) in results.iter().enumerate() {
        let (a, b) = match (rt, interp[i]) {
            (svm_ir::ValType::I32, Value::I32(x)) => (x as u32 as u64, jit[i] as u32 as u64),
            (svm_ir::ValType::I64, Value::I64(x)) => (x as u64, jit[i] as u64),
            _ => panic!("result type / value mismatch at {i}"),
        };
        assert_eq!(a, b, "interp != jit at result {i}");
    }
    match interp[0] {
        Value::I64(x) => x,
        Value::I32(x) => x as i64,
        other => panic!("unexpected interp value {other:?}"),
    }
}

/// Like [`run`] but for any value type (incl. floats): runs interp + JIT, asserts they agree
/// (float results bit-equal or both NaN), and returns the interp result `Value`.
fn eval(wat: &str, entry: &str, args: &[Value]) -> Value {
    let wasm = wat::parse_str(wat).expect("assemble wat");
    let t = svm_wasm::transpile(&wasm).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify");
    let idx = t.exports.iter().find(|(n, _)| n == entry).unwrap().1;
    let results = &t.module.funcs[idx as usize].results;
    let mut fuel = 100_000_000u64;
    let interp = svm_interp::run(&t.module, idx, args, &mut fuel).expect("interp");
    let jit_args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(x) => *x as i64,
            Value::I64(x) => *x,
            Value::F32(x) => x.to_bits() as i64,
            Value::F64(x) => x.to_bits() as i64,
        })
        .collect();
    let jit = match svm_jit::compile_and_run(&t.module, idx, &jit_args).expect("jit") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    for (i, rt) in results.iter().enumerate() {
        let ok = match (rt, interp[i]) {
            (svm_ir::ValType::I32, Value::I32(x)) => x as u32 as u64 == jit[i] as u32 as u64,
            (svm_ir::ValType::I64, Value::I64(x)) => x as u64 == jit[i] as u64,
            (svm_ir::ValType::F32, Value::F32(x)) => {
                let j = f32::from_bits(jit[i] as u32);
                x.to_bits() == j.to_bits() || (x.is_nan() && j.is_nan())
            }
            (svm_ir::ValType::F64, Value::F64(x)) => {
                let j = f64::from_bits(jit[i] as u64);
                x.to_bits() == j.to_bits() || (x.is_nan() && j.is_nan())
            }
            _ => panic!("result type/value mismatch"),
        };
        assert!(
            ok,
            "interp != jit at result {i}: {:?} vs {:#x}",
            interp[i], jit[i]
        );
    }
    interp[0]
}

fn as_f64(v: Value) -> f64 {
    match v {
        Value::F64(x) => x,
        other => panic!("expected f64, got {other:?}"),
    }
}
fn as_f32(v: Value) -> f32 {
    match v {
        Value::F32(x) => x,
        other => panic!("expected f32, got {other:?}"),
    }
}

#[test]
fn f64_arithmetic() {
    let wat = r#"
(module (func (export "f") (param $a f64) (param $b f64) (result f64)
  (f64.add (f64.mul (local.get $a) (local.get $a)) (f64.sqrt (local.get $b)))))"#;
    assert_eq!(
        as_f64(eval(wat, "f", &[Value::F64(3.0), Value::F64(16.0)])),
        13.0
    );
}

/// A float loop: sum 1/k for k in 1..=n (harmonic), plus int↔float conversion — exercises FBin/FCmp,
/// the loop, and i64→f64 / f64 compares.
#[test]
fn f64_harmonic_loop() {
    let wat = r#"
(module (func (export "h") (param $n i64) (result f64)
  (local $acc f64) (local $k i64)
  (local.set $k (i64.const 1))
  (block $done (loop $loop
    (br_if $done (i64.gt_s (local.get $k) (local.get $n)))
    (local.set $acc (f64.add (local.get $acc) (f64.div (f64.const 1) (f64.convert_i64_s (local.get $k)))))
    (local.set $k (i64.add (local.get $k) (i64.const 1)))
    (br $loop)))
  (local.get $acc)))"#;
    let got = as_f64(eval(wat, "h", &[Value::I64(4)]));
    let want = 1.0 + 0.5 + 1.0 / 3.0 + 0.25;
    assert!(
        (got - want).abs() < 1e-12,
        "harmonic(4) = {got}, want {want}"
    );
}

#[test]
fn f32_and_conversions() {
    let wat = r#"
(module (func (export "g") (param $x f32) (result i32)
  (i32.trunc_f32_s (f32.mul (local.get $x) (f32.const 2.5)))))"#;
    assert_eq!(eval(wat, "g", &[Value::F32(4.0)]), Value::I32(10));
    // demote/promote round trip
    let wat2 = r#"
(module (func (export "rt") (param $x f64) (result f64)
  (f64.promote_f32 (f32.demote_f64 (local.get $x)))))"#;
    let got = as_f32(eval(
        r#"(module (func (export "d") (param $x f64) (result f32) (f32.demote_f64 (local.get $x))))"#,
        "d",
        &[Value::F64(1.5)],
    ));
    assert_eq!(got, 1.5f32);
    assert_eq!(as_f64(eval(wat2, "rt", &[Value::F64(2.25)])), 2.25);
}

#[test]
fn straight_line_add() {
    let wat = r#"
(module (func (export "add") (param i32 i32) (result i32)
  (i32.add (local.get 0) (local.get 1))))"#;
    assert_eq!(run(wat, "add", &[Value::I32(2), Value::I32(3)]), 5);
    assert_eq!(
        run(wat, "add", &[Value::I32(i32::MAX), Value::I32(1)]),
        i32::MIN as i64
    ); // wraps, like our IR
}

#[test]
fn locals_and_arithmetic() {
    // r = (a*a + b) ; tee/get/set exercised
    let wat = r#"
(module (func (export "f") (param $a i64) (param $b i64) (result i64)
  (local $t i64)
  (local.set $t (i64.mul (local.get $a) (local.get $a)))
  (i64.add (local.get $t) (local.get $b))))"#;
    assert_eq!(run(wat, "f", &[Value::I64(7), Value::I64(5)]), 54);
}

/// The actual `alu` benchmark kernel: an LCG recurrence in a `block`/`loop` with `br_if`/`br` — the
/// first real proof the stack→SSA + control-flow lowering produces correct code.
#[test]
fn alu_lcg_loop() {
    let wat = r#"
(module
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
        (local.set $acc
          (i64.add
            (i64.add
              (i64.mul (local.get $acc) (i64.const 6364136223846793005))
              (i64.const 1442695040888963407))
            (local.get $i)))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))"#;
    for n in [0i64, 1, 2, 5, 10, 37] {
        let got = run(wat, "run", &[Value::I64(n)]);
        assert_eq!(got, alu_ref(n), "alu mismatch at n={n}");
    }
}

/// Reference LCG: `acc = acc*C1 + C2 + i` for i in 0..n (wrapping i64).
fn alu_ref(n: i64) -> i64 {
    let mut acc: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        acc = acc
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            .wrapping_add(i);
        i += 1;
    }
    acc
}

/// Nested loop + early break via `br` to an outer block — exercises multi-level control + br_table.
#[test]
fn br_table_dispatch() {
    // returns [10,20,30][sel], or 99 for out-of-range (the default).
    let wat = r#"
(module (func (export "pick") (param $sel i32) (result i32)
  (block $b3 (block $b2 (block $b1 (block $b0
    (br_table $b0 $b1 $b2 $b3 (local.get $sel)))
    (return (i32.const 10)))
    (return (i32.const 20)))
    (return (i32.const 30)))
  (i32.const 99)))"#;
    assert_eq!(run(wat, "pick", &[Value::I32(0)]), 10);
    assert_eq!(run(wat, "pick", &[Value::I32(1)]), 20);
    assert_eq!(run(wat, "pick", &[Value::I32(2)]), 30);
    assert_eq!(run(wat, "pick", &[Value::I32(7)]), 99);
}

#[test]
fn if_else_max() {
    let wat = r#"
(module (func (export "max") (param $a i32) (param $b i32) (result i32)
  (if (result i32) (i32.gt_s (local.get $a) (local.get $b))
    (then (local.get $a)) (else (local.get $b)))))"#;
    assert_eq!(run(wat, "max", &[Value::I32(7), Value::I32(3)]), 7);
    assert_eq!(run(wat, "max", &[Value::I32(3), Value::I32(9)]), 9);
    assert_eq!(run(wat, "max", &[Value::I32(-5), Value::I32(-2)]), -2);
}

/// `if` without `else` (the inputs/locals pass through): clamp negatives to zero via a side-effecting
/// then arm. Exercises the implicit pass-through else.
#[test]
fn if_no_else_clamp() {
    let wat = r#"
(module (func (export "clamp") (param $x i32) (result i32)
  (local $r i32)
  (local.set $r (local.get $x))
  (if (i32.lt_s (local.get $x) (i32.const 0)) (then (local.set $r (i32.const 0))))
  (local.get $r)))"#;
    assert_eq!(run(wat, "clamp", &[Value::I32(5)]), 5);
    assert_eq!(run(wat, "clamp", &[Value::I32(-5)]), 0);
    assert_eq!(run(wat, "clamp", &[Value::I32(0)]), 0);
}

/// The then arm `br`s out of an enclosing block (going dead), so the **else arm must still be
/// reachable** — the dead-then / else-resurrection path.
#[test]
fn if_then_br_else_resurrects() {
    let wat = r#"
(module (func (export "g") (param $c i32) (result i32)
  (block $b (result i32)
    (if (result i32) (local.get $c)
      (then (br $b (i32.const 1)))
      (else (i32.const 2))))))"#;
    assert_eq!(run(wat, "g", &[Value::I32(1)]), 1);
    assert_eq!(run(wat, "g", &[Value::I32(0)]), 2);
}

/// Nested if/else inside a loop — collatz step count, exercising if/else + loop + br interplay.
#[test]
fn collatz_steps() {
    let wat = r#"
(module (func (export "steps") (param $n i64) (result i64)
  (local $c i64)
  (block $done (loop $loop
    (br_if $done (i64.le_s (local.get $n) (i64.const 1)))
    (if (i64.eqz (i64.rem_u (local.get $n) (i64.const 2)))
      (then (local.set $n (i64.div_u (local.get $n) (i64.const 2))))
      (else (local.set $n (i64.add (i64.mul (local.get $n) (i64.const 3)) (i64.const 1)))))
    (local.set $c (i64.add (local.get $c) (i64.const 1)))
    (br $loop)))
  (local.get $c)))"#;
    // 6 → 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1 : 8 steps
    assert_eq!(run(wat, "steps", &[Value::I64(6)]), 8);
    assert_eq!(run(wat, "steps", &[Value::I64(1)]), 0);
    assert_eq!(run(wat, "steps", &[Value::I64(27)]), 111);
}

#[test]
fn memory_store_load_roundtrip() {
    let wat = r#"
(module (memory 1)
  (func (export "rw") (param $a i32) (param $v i64) (result i64)
    (i64.store (local.get $a) (local.get $v))
    (i64.load (local.get $a))))"#;
    assert_eq!(
        run(wat, "rw", &[Value::I32(80), Value::I64(123456789)]),
        123456789
    );
    // narrow store/load truncates like wasm
    let wat8 = r#"
(module (memory 1)
  (func (export "rw8") (param $a i32) (param $v i32) (result i32)
    (i32.store8 (local.get $a) (local.get $v))
    (i32.load8_u (local.get $a))))"#;
    assert_eq!(run(wat8, "rw8", &[Value::I32(16), Value::I32(0x1ff)]), 0xff);
}

/// The real `memsum` bench kernel (wasm32): store `i` to a windowed slot, read it back, sum. Each slot
/// is overwritten then read in the same iteration, so the total is `Σ i = n(n-1)/2`.
#[test]
fn memsum_kernel_wasm32() {
    let wat = r#"
(module (memory 1)
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64) (local $addr i32)
    (block $done (loop $loop
      (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
      (local.set $addr (i32.mul (i32.and (i32.wrap_i64 (local.get $i)) (i32.const 1023)) (i32.const 8)))
      (i64.store (local.get $addr) (local.get $i))
      (local.set $acc (i64.add (local.get $acc) (i64.load (local.get $addr))))
      (local.set $i (i64.add (local.get $i) (i64.const 1)))
      (br $loop)))
    (local.get $acc)))"#;
    for n in [0i64, 1, 10, 100] {
        assert_eq!(run(wat, "run", &[Value::I64(n)]), n * (n - 1) / 2);
    }
}

/// Same kernel over a **64-bit** memory (`memory i64`) — the address is already i64, no extension.
#[test]
fn memsum_kernel_wasm64() {
    let wat = r#"
(module (memory i64 1)
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64) (local $addr i64)
    (block $done (loop $loop
      (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
      (local.set $addr (i64.mul (i64.and (local.get $i) (i64.const 1023)) (i64.const 8)))
      (i64.store (local.get $addr) (local.get $i))
      (local.set $acc (i64.add (local.get $acc) (i64.load (local.get $addr))))
      (local.set $i (i64.add (local.get $i) (i64.const 1)))
      (br $loop)))
    (local.get $acc)))"#;
    for n in [0i64, 1, 10, 100] {
        assert_eq!(run(wat, "run", &[Value::I64(n)]), n * (n - 1) / 2);
    }
}

/// The `scatter` kernel: store to one hashed slot, load from a different one — addresses that vary per
/// iteration, with the array persisting across iterations. Checked against a Rust replica.
#[test]
fn scatter_kernel() {
    let wat = r#"
(module (memory 1)
  (func (export "run") (param $n i64) (result i64)
    (local $acc i64) (local $i i64)
    (block $done (loop $loop
      (br_if $done (i64.ge_s (local.get $i) (local.get $n)))
      (i64.store
        (i32.mul (i32.and (i32.wrap_i64 (i64.mul (local.get $i) (i64.const 2654435761))) (i32.const 1023)) (i32.const 8))
        (local.get $i))
      (local.set $acc (i64.add (local.get $acc)
        (i64.load
          (i32.mul (i32.and (i32.wrap_i64 (i64.mul (local.get $i) (i64.const 2246822519))) (i32.const 1023)) (i32.const 8)))))
      (local.set $i (i64.add (local.get $i) (i64.const 1)))
      (br $loop)))
    (local.get $acc)))"#;
    for n in [0i64, 1, 5, 50, 300] {
        assert_eq!(
            run(wat, "run", &[Value::I64(n)]),
            scatter_ref(n),
            "scatter n={n}"
        );
    }
}

fn scatter_ref(n: i64) -> i64 {
    let mut mem = [0i64; 1024];
    let mut acc = 0i64;
    for i in 0..n {
        let si = ((i.wrapping_mul(2654435761) as i32) & 1023) as usize;
        mem[si] = i;
        let li = ((i.wrapping_mul(2246822519) as i32) & 1023) as usize;
        acc = acc.wrapping_add(mem[li]);
    }
    acc
}

#[test]
fn direct_call_multifunction() {
    let wat = r#"
(module
  (func $sq (param $x i64) (result i64) (i64.mul (local.get $x) (local.get $x)))
  (func (export "sumsq") (param $a i64) (param $b i64) (result i64)
    (i64.add (call $sq (local.get $a)) (call $sq (local.get $b)))))"#;
    assert_eq!(run(wat, "sumsq", &[Value::I64(3), Value::I64(4)]), 25);
    assert_eq!(run(wat, "sumsq", &[Value::I64(-5), Value::I64(12)]), 169);
}

/// Recursion through `call` (Fibonacci) — exercises call + if/else + the call stack.
#[test]
fn recursive_call_fib() {
    let wat = r#"
(module (func $fib (export "fib") (param $n i64) (result i64)
  (if (result i64) (i64.lt_s (local.get $n) (i64.const 2))
    (then (local.get $n))
    (else (i64.add (call $fib (i64.sub (local.get $n) (i64.const 1)))
                   (call $fib (i64.sub (local.get $n) (i64.const 2))))))))"#;
    for (n, want) in [(0i64, 0i64), (1, 1), (10, 55), (20, 6765)] {
        assert_eq!(run(wat, "fib", &[Value::I64(n)]), want, "fib({n})");
    }
}

/// An active data segment initializes linear memory; the guest reads it back.
#[test]
fn data_segment_init() {
    let wat = r#"
(module (memory 1)
  (data (i32.const 16) "\01\02\03\04\05\06\07\08")
  (func (export "g") (result i64) (i64.load (i32.const 16))))"#;
    // little-endian i64 from bytes 01..08
    assert_eq!(run(wat, "g", &[]), 0x0807_0605_0403_0201);

    // sum two i32s laid out by a data segment
    let wat2 = r#"
(module (memory 1)
  (data (i32.const 0) "\0a\00\00\00\14\00\00\00")
  (func (export "sum") (result i32)
    (i32.add (i32.load (i32.const 0)) (i32.load (i32.const 4)))))"#;
    assert_eq!(run(wat2, "sum", &[]), 30); // 10 + 20
}

/// **Capstone: real clang-emitted wasm.** Compile C to wasm with `clang --target=wasm32` (+ `wasm-ld`)
/// and run the transpiled module — exercising LLVM-optimized control flow, the `__stack_pointer`
/// mutable global, and data layout that no hand-written WAT here covers. Skipped (not failed) if the
/// wasm toolchain is unavailable, matching how the C-frontend tests treat a missing `cc`.
#[cfg(unix)]
#[test]
fn real_clang_wasm() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("svm_wasm_clang_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c = dir.join("t.c");
    let w = dir.join("t.wasm");
    std::fs::write(
        &c,
        "int fib(int n){return n<2?n:fib(n-1)+fib(n-2);}\n\
         int sumto(int n){int s=0;for(int i=1;i<=n;i++)s+=i;return s;}\n\
         int poly(int x){return 3*x*x - 5*x + 7;}\n\
         static int add(int a,int b){return a+b;}\n\
         static int sub(int a,int b){return a-b;}\n\
         static int mul(int a,int b){return a*b;}\n\
         int dispatch(int op,int a,int b){\n\
           int (*tbl[3])(int,int)={add,sub,mul}; return tbl[op](a,b);}\n",
    )
    .unwrap();
    let out = Command::new("clang")
        .args(["--target=wasm32", "-nostdlib", "-O2"])
        .args([
            "-Wl,--no-entry",
            "-Wl,--export=fib",
            "-Wl,--export=sumto",
            "-Wl,--export=poly",
            "-Wl,--export=dispatch",
        ])
        .arg(&c)
        .arg("-o")
        .arg(&w)
        .output();
    match out {
        Ok(o) if o.status.success() => {}
        _ => {
            eprintln!("skipping real_clang_wasm: clang wasm toolchain unavailable");
            return;
        }
    }
    let wasm = std::fs::read(&w).unwrap();
    let t = svm_wasm::transpile(&wasm).expect("transpile real clang wasm");
    svm_verify::verify_module(&t.module).expect("verify real clang wasm");
    let find = |name: &str| t.exports.iter().find(|(n, _)| n == name).unwrap().1;
    assert_eq!(run_idx(&t, find("fib"), &[Value::I32(20)]), 6765);
    assert_eq!(run_idx(&t, find("sumto"), &[Value::I32(100)]), 5050);
    for x in [0i32, 3, -4, 17] {
        assert_eq!(
            run_idx(&t, find("poly"), &[Value::I32(x)]),
            (3 * x * x - 5 * x + 7) as i64,
            "poly({x})"
        );
    }
    // `dispatch` is a C function-pointer table — clang lowers it to call_indirect + a table + an
    // element segment, exercising the whole indirect-call path on real-world wasm.
    let d = find("dispatch");
    assert_eq!(
        run_idx(&t, d, &[Value::I32(0), Value::I32(7), Value::I32(3)]),
        10
    );
    assert_eq!(
        run_idx(&t, d, &[Value::I32(1), Value::I32(7), Value::I32(3)]),
        4
    );
    assert_eq!(
        run_idx(&t, d, &[Value::I32(2), Value::I32(7), Value::I32(3)]),
        21
    );
}

/// Run a known function index through interp + JIT, assert they agree, return the (i32/i64) result.
/// Only used by the `#[cfg(unix)]` `real_clang_wasm` test, so gate it the same way (else it's dead
/// code on non-unix targets, which CI's `-D warnings` rejects).
#[cfg(unix)]
fn run_idx(t: &svm_wasm::Transpiled, idx: u32, args: &[Value]) -> i64 {
    let rt = t.module.funcs[idx as usize].results[0];
    let mut fuel = 100_000_000u64;
    let interp = svm_interp::run(&t.module, idx, args, &mut fuel).expect("interp");
    let jit_args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(x) => *x as i64,
            Value::I64(x) => *x,
            _ => panic!(),
        })
        .collect();
    let jit = match svm_jit::compile_and_run(&t.module, idx, &jit_args).unwrap() {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    match (rt, interp[0]) {
        (svm_ir::ValType::I32, Value::I32(x)) => {
            assert_eq!(x as u32, jit[0] as u32, "interp != jit");
            x as i64
        }
        (svm_ir::ValType::I64, Value::I64(x)) => {
            assert_eq!(x as u64, jit[0] as u64, "interp != jit");
            x
        }
        _ => panic!("unexpected result"),
    }
}

/// A mutable global used as accumulator state across get/set (with linear memory present).
#[test]
fn mutable_global_counter() {
    let wat = r#"
(module (memory 1)
  (global $g (mut i32) (i32.const 100))
  (func (export "f") (param $x i32) (result i32)
    (global.set $g (i32.add (global.get $g) (local.get $x)))
    (global.get $g)))"#;
    // each run re-instantiates (the data segment re-inits the global to 100)
    assert_eq!(run(wat, "f", &[Value::I32(5)]), 105);
    assert_eq!(run(wat, "f", &[Value::I32(-30)]), 70);
}

/// A module with **globals but no linear memory** — the transpiler still gives them a window region.
#[test]
fn globals_without_memory() {
    let wat = r#"
(module
  (global $g (mut i64) (i64.const 7))
  (func (export "acc") (param $x i64) (result i64)
    (global.set $g (i64.mul (global.get $g) (local.get $x)))
    (global.get $g)))"#;
    assert_eq!(run(wat, "acc", &[Value::I64(3)]), 21);
    assert_eq!(run(wat, "acc", &[Value::I64(6)]), 42);
}

/// Immutable + float globals.
#[test]
fn immutable_and_float_globals() {
    let wat = r#"
(module
  (global $c i32 (i32.const 42))
  (global $pi f64 (f64.const 3.25))
  (func (export "c") (result i32) (global.get $c))
  (func (export "twopi") (result f64) (f64.add (global.get $pi) (global.get $pi))))"#;
    assert_eq!(run(wat, "c", &[]), 42);
    assert_eq!(as_f64(eval(wat, "twopi", &[])), 6.5);
}

/// `call_indirect` through a function table populated by an element segment — virtual dispatch.
#[test]
fn call_indirect_dispatch() {
    let wat = r#"
(module
  (table 3 funcref)
  (elem (i32.const 0) $add $sub $mul)
  (type $binop (func (param i32 i32) (result i32)))
  (func $add (type $binop) (i32.add (local.get 0) (local.get 1)))
  (func $sub (type $binop) (i32.sub (local.get 0) (local.get 1)))
  (func $mul (type $binop) (i32.mul (local.get 0) (local.get 1)))
  (func (export "dispatch") (param $op i32) (param $a i32) (param $b i32) (result i32)
    (call_indirect (type $binop) (local.get $a) (local.get $b) (local.get $op))))"#;
    assert_eq!(
        run(
            wat,
            "dispatch",
            &[Value::I32(0), Value::I32(7), Value::I32(3)]
        ),
        10
    );
    assert_eq!(
        run(
            wat,
            "dispatch",
            &[Value::I32(1), Value::I32(7), Value::I32(3)]
        ),
        4
    );
    assert_eq!(
        run(
            wat,
            "dispatch",
            &[Value::I32(2), Value::I32(7), Value::I32(3)]
        ),
        21
    );
}

/// A `call_indirect` whose declared type doesn't match the table entry's must **trap** (the §3c
/// type-id check), on both backends — the I2 "forged/confused index is inert" guarantee.
#[test]
fn call_indirect_type_mismatch_traps() {
    let wat = r#"
(module
  (table 1 funcref)
  (elem (i32.const 0) $f)
  (type $unary (func (param i64) (result i64)))
  (func $f (param i32 i32) (result i32) (i32.add (local.get 0) (local.get 1)))
  (func (export "bad") (result i64)
    (call_indirect (type $unary) (i64.const 5) (i32.const 0))))"#;
    let wasm = wat::parse_str(wat).unwrap();
    let t = svm_wasm::transpile(&wasm).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify");
    let idx = t.exports.iter().find(|(n, _)| n == "bad").unwrap().1;
    let mut fuel = 1_000_000u64;
    assert!(
        svm_interp::run(&t.module, idx, &[], &mut fuel).is_err(),
        "interp must trap"
    );
    assert!(
        matches!(
            svm_jit::compile_and_run(&t.module, idx, &[]).unwrap(),
            svm_jit::JitOutcome::Trapped(_)
        ),
        "jit must trap"
    );
}

/// `memory.size` with no growth is the constant initial page count (no runtime cell needed).
#[test]
fn memory_size_constant() {
    let wat = r#"(module (memory 3) (func (export "sz") (result i32) (memory.size)))"#;
    assert_eq!(run(wat, "sz", &[]), 3);
}

/// `memory.grow` returns the previous size, and `memory.size` then reflects the larger memory (the
/// runtime size cell is read back).
#[test]
fn memory_grow_returns_old_and_updates_size() {
    let old = r#"(module (memory 1) (func (export "g") (result i32) (memory.grow (i32.const 2))))"#;
    assert_eq!(run(old, "g", &[]), 1); // previous size in pages

    let sz = r#"
(module (memory 1)
  (func (export "sz") (result i32)
    (drop (memory.grow (i32.const 2)))
    (memory.size)))"#;
    assert_eq!(run(sz, "sz", &[]), 3); // 1 + 2 pages
}

/// A `memory.grow` past the cap (unbounded memory's default `DEFAULT_MAX_GROW_PAGES = 256`) returns
/// `-1` and leaves the size unchanged.
#[test]
fn memory_grow_over_cap_fails() {
    let r =
        r#"(module (memory 1) (func (export "g") (result i32) (memory.grow (i32.const 1000))))"#;
    assert_eq!(run(r, "g", &[]), -1);

    let sz = r#"
(module (memory 1)
  (func (export "sz") (result i32)
    (drop (memory.grow (i32.const 1000)))
    (memory.size)))"#;
    assert_eq!(run(sz, "sz", &[]), 1); // unchanged
}

/// A declared `maximum` is honored as the grow cap (rather than the unbounded default): growing to the
/// max succeeds, one past it fails.
#[test]
fn memory_grow_honors_declared_maximum() {
    let ok =
        r#"(module (memory 1 4) (func (export "g") (result i32) (memory.grow (i32.const 3))))"#;
    assert_eq!(run(ok, "g", &[]), 1); // 1 -> 4 (== maximum) succeeds, returns old size
    let fail =
        r#"(module (memory 1 4) (func (export "g") (result i32) (memory.grow (i32.const 4))))"#;
    assert_eq!(run(fail, "g", &[]), -1); // 1 -> 5 (> maximum) fails
}

/// After growing, the new pages are usable — a store/load to an address in the grown region (past the
/// initial 64 KiB) round-trips identically on both backends (the window holds the growable span).
#[test]
fn grown_memory_is_usable() {
    let wat = r#"
(module (memory 1)
  (func (export "g") (result i64)
    (drop (memory.grow (i32.const 1)))      ;; 1 -> 2 pages (128 KiB)
    (i64.store (i32.const 70000) (i64.const 0x0102030405060708))
    (i64.load (i32.const 70000))))"#;
    assert_eq!(run(wat, "g", &[]), 0x0102030405060708);
}

/// `memory64`: `memory.size`/`memory.grow` operate in i64.
#[test]
fn memory64_grow_and_size() {
    let g =
        r#"(module (memory i64 1) (func (export "g") (result i64) (memory.grow (i64.const 2))))"#;
    assert_eq!(run(g, "g", &[]), 1);
    let sz = r#"
(module (memory i64 1)
  (func (export "sz") (result i64)
    (drop (memory.grow (i64.const 2)))
    (memory.size)))"#;
    assert_eq!(run(sz, "sz", &[]), 3);
}

#[test]
fn unsupported_is_clean_error() {
    // A **non-constant-length** `memory.fill` is out of the current subset (only constant sizes unroll)
    // → a clean Unsupported error, not a panic. (A SIMD/passive-segment op would do equally.)
    let wat = r#"(module (memory 1)
      (func (export "f") (param $n i32) (memory.fill (i32.const 0) (i32.const 0) (local.get $n))))"#;
    let wasm = wat::parse_str(wat).unwrap();
    match svm_wasm::transpile(&wasm) {
        Err(svm_wasm::Error::Unsupported(_)) => {}
        Err(e) => panic!("expected Unsupported, got error {e:?}"),
        Ok(_) => panic!("expected Unsupported, got Ok"),
    }
}

/// Hand-written `memory.copy` over **overlapping** ranges — exercises the memmove semantics (load all
/// before storing any). `data` seeds 0..8 at offset 0; copy 6 bytes from 0 to 2 (overlap), so
/// `[2..8] = [0,1,2,3,4,5]`. Reading byte at `idx` proves the overlap-correct result on both backends.
#[test]
fn memory_copy_overlap_is_memmove() {
    let wat = r#"
(module (memory 1)
  (data (i32.const 0) "\00\01\02\03\04\05\06\07")
  (func (export "byte") (param $idx i32) (result i32)
    (memory.copy (i32.const 2) (i32.const 0) (i32.const 6))   ;; dest=2, src=0, len=6 (overlap)
    (i32.load8_u (local.get $idx))))"#;
    // After memmove: bytes = [0,1,0,1,2,3,4,5]. (A naive forward byte loop would give [0,1,0,1,0,1,..].)
    let expect = [0, 1, 0, 1, 2, 3, 4, 5];
    for (i, &e) in expect.iter().enumerate() {
        assert_eq!(run(wat, "byte", &[Value::I32(i as i32)]), e, "byte[{i}]");
    }
}

/// Hand-written `memory.fill` — set a run of bytes to a value, read one back. Exercises the broadcast
/// chunking (8/4/2/1) at a non-byte-multiple length.
#[test]
fn memory_fill_sets_bytes() {
    let wat = r#"
(module (memory 1)
  (func (export "byte") (param $idx i32) (result i32)
    (memory.fill (i32.const 4) (i32.const 0xAB) (i32.const 13))  ;; [4..17) = 0xAB
    (i32.load8_u (local.get $idx))))"#;
    assert_eq!(run(wat, "byte", &[Value::I32(3)]), 0); // before the fill
    for i in 4..17 {
        assert_eq!(run(wat, "byte", &[Value::I32(i)]), 0xAB, "byte[{i}]");
    }
    assert_eq!(run(wat, "byte", &[Value::I32(17)]), 0); // after the fill
}

/// **Real clang program using bulk memory.** A struct copy by value (clang → `memory.copy`) and a
/// large zero-init (`int buf[64]={0}` → `memory.fill`), compiled with `-mbulk-memory`, transpiled, and
/// run on interp + JIT against a hand-computed oracle — the program-first proof that real bulk-memory
/// wasm runs identically.
#[cfg(unix)]
#[test]
fn real_clang_bulk_memory() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("svm_wasm_bulk_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c = dir.join("b.c");
    let w = dir.join("b.wasm");
    std::fs::write(
        &c,
        "struct Big { int a[24]; };\n\
         static struct Big g_src;\n\
         int sum_copy(int n){\n\
           struct Big x; for(int i=0;i<24;i++) x.a[i]=i*n+1;\n\
           struct Big y = x; g_src = y;\n\
           int s=0; for(int i=0;i<24;i++) s+=g_src.a[i]; return s; }\n\
         int zero_then_set(int n){\n\
           int buf[64]={0}; buf[n&63]=99;\n\
           int s=0; for(int i=0;i<64;i++) s+=buf[i]; return s; }\n",
    )
    .unwrap();
    let out = Command::new("clang")
        .args(["--target=wasm32", "-nostdlib", "-O2", "-mbulk-memory"])
        .args([
            "-Wl,--no-entry",
            "-Wl,--export=sum_copy",
            "-Wl,--export=zero_then_set",
        ])
        .arg(&c)
        .arg("-o")
        .arg(&w)
        .output();
    match out {
        Ok(o) if o.status.success() => {}
        _ => {
            eprintln!("skipping real_clang_bulk_memory: clang wasm toolchain unavailable");
            return;
        }
    }
    let wasm = std::fs::read(&w).unwrap();
    let t = svm_wasm::transpile(&wasm).expect("transpile bulk-memory wasm");
    svm_verify::verify_module(&t.module).expect("verify");
    let find = |name: &str| t.exports.iter().find(|(n, _)| n == name).unwrap().1;
    // sum_copy(n) = Σ_{i=0..23}(i·n+1) = 276·n + 24.
    for n in [0i32, 1, 2, 5] {
        assert_eq!(
            run_idx(&t, find("sum_copy"), &[Value::I32(n)]),
            (276 * n + 24) as i64,
            "sum_copy({n})"
        );
    }
    // zero_then_set: one element set to 99, the rest zero ⇒ 99 regardless of n.
    for n in [0i32, 7, 63, 100] {
        assert_eq!(run_idx(&t, find("zero_then_set"), &[Value::I32(n)]), 99);
    }
}
