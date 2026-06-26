//! In-process Wasmtime (Cranelift) timer for **one** Embench cross-engine kernel — the wasmtime
//! counterpart to `bench/embench/wasm/run.mjs` (V8). Loads a self-contained module exporting
//! `run(long n)` (no imports), warms it up so Cranelift compiles at instantiate, then prints one
//! per-iteration time and the verify result, same methodology as the native/SVM/V8 drivers:
//! per_iter = (min t(large) - min t(small)) / (large - small), min over reps.
//!
//!   embench_one <kernel.wasm> <small> <large> <verify_n>
//! stdout: two lines — "<per_iter_ns>" then "<verify>" (matches the harness's parse).
//!
//! Handles both **wasm32** and **wasm64** (memory64) modules: `wasm_memory64(true)` is always set
//! (permissive — 32-bit modules still load), and the `run` argument/return width (i32 vs i64) is
//! auto-detected from the export's signature, since under wasm64 `long` is 64-bit so `run` is i64(i64).
use std::time::Instant;
use wasmtime::{Config, Engine, Instance, Module, Store, Val};

const REPS: usize = 10;

/// `run`'s param is i64 under wasm64 (`long` is 64-bit there) and i32 under wasm32.
fn is64(store: &Store<()>, f: &wasmtime::Func) -> bool {
    matches!(f.ty(store).params().next(), Some(wasmtime::ValType::I64))
}

fn arg(n: i64, wide: bool) -> Val {
    if wide {
        Val::I64(n)
    } else {
        Val::I32(n as i32)
    }
}

fn ret(out: &Val) -> i64 {
    match out {
        Val::I64(x) => *x,
        Val::I32(x) => *x as i64,
        other => panic!("unexpected return {other:?}"),
    }
}

fn best(store: &mut Store<()>, f: &wasmtime::Func, n: i64, wide: bool) -> f64 {
    let mut out = [Val::I32(0)];
    f.call(&mut *store, &[arg(n, wide)], &mut out).unwrap(); // warm up (compile at instantiate)
    let mut b = f64::MAX;
    for _ in 0..REPS {
        let t = Instant::now();
        f.call(&mut *store, &[arg(n, wide)], &mut out).unwrap();
        b = b.min(t.elapsed().as_nanos() as f64);
    }
    b
}

fn call1(store: &mut Store<()>, f: &wasmtime::Func, n: i64, wide: bool) -> i64 {
    let mut out = [Val::I32(0)];
    f.call(&mut *store, &[arg(n, wide)], &mut out).unwrap();
    ret(&out[0])
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, small, large, vn): (&str, i64, i64, i64) = (
        &a[1],
        a[2].parse().unwrap(),
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
    );
    let mut cfg = Config::new();
    cfg.wasm_memory64(true); // permissive: also loads plain wasm32 modules
    let engine = Engine::new(&cfg).unwrap();
    let module = Module::from_file(&engine, path).unwrap();
    let mut store = Store::new(&engine, ());
    let inst = Instance::new(&mut store, &module, &[]).unwrap();
    let f = inst.get_func(&mut store, "run").expect("`run` export");
    let wide = is64(&store, &f);
    let s = best(&mut store, &f, small, wide);
    let l = best(&mut store, &f, large, wide);
    println!("{:.6}", (l - s) / (large - small) as f64);
    println!("{}", call1(&mut store, &f, vn, wide));
}
