//! In-process Wasmtime (Cranelift) timer for **one** Embench cross-engine kernel — the wasmtime
//! counterpart to `bench/embench/wasm/run.mjs` (V8). Loads a self-contained wasm32 module exporting
//! `run(long n)` (no imports), warms it up so Cranelift compiles at instantiate, then prints one
//! per-iteration time and the verify result, same methodology as the native/SVM/V8 drivers:
//! per_iter = (min t(large) - min t(small)) / (large - small), min over reps.
//!
//!   embench_one <kernel.wasm> <small> <large> <verify_n>
//! stdout: two lines — "<per_iter_ns>" then "<verify>" (matches the harness's parse).
use std::time::Instant;
use wasmtime::{Config, Engine, Instance, Module, Store, Val};

const REPS: usize = 10;

fn best(store: &mut Store<()>, f: &wasmtime::Func, n: i64) -> f64 {
    let mut out = [Val::I32(0)];
    f.call(&mut *store, &[Val::I32(n as i32)], &mut out)
        .unwrap(); // warm up (compile at instantiate)
    let mut b = f64::MAX;
    for _ in 0..REPS {
        let t = Instant::now();
        f.call(&mut *store, &[Val::I32(n as i32)], &mut out)
            .unwrap();
        b = b.min(t.elapsed().as_nanos() as f64);
    }
    b
}

fn call1(store: &mut Store<()>, f: &wasmtime::Func, n: i64) -> i64 {
    let mut out = [Val::I32(0)];
    f.call(&mut *store, &[Val::I32(n as i32)], &mut out)
        .unwrap();
    out[0].unwrap_i32() as i64
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, small, large, vn): (&str, i64, i64, i64) = (
        &a[1],
        a[2].parse().unwrap(),
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
    );
    let engine = Engine::new(&Config::new()).unwrap();
    let module = Module::from_file(&engine, path).unwrap();
    let mut store = Store::new(&engine, ());
    let inst = Instance::new(&mut store, &module, &[]).unwrap();
    let f = inst.get_func(&mut store, "run").expect("`run` export");
    let s = best(&mut store, &f, small);
    let l = best(&mut store, &f, large);
    println!("{:.6}", (l - s) / (large - small) as f64);
    println!("{}", call1(&mut store, &f, vn));
}
