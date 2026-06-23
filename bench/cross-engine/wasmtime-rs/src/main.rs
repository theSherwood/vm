//! In-process Wasmtime (Cranelift) timer — the accurate counterpart to the `wasmtime_bench.py` CLI
//! driver. Times each kernel in-process (no per-process spawn/compile overhead), so it resolves even
//! the ~0.1 ns/iter `vsum` and is directly comparable to the in-process V8 numbers. Same methodology:
//! per-iteration = (min time at n=201000 − min time at n=1000) / 200000, min over reps.
//! Usage: wasmtime-bench <k32.wasm> <k64.wasm>
use std::time::Instant;
use wasmtime::{Config, Engine, Instance, Module, Store, Val};

const NS: i32 = 1_000;
const NL: i32 = 201_000;
const REPS: usize = 25;

// Every kernel is i32(i32) now.
const KERNELS: &[&str] = &[
    "alu",
    "xorshift",
    "call",
    "call_indirect",
    "mem",
    "chase",
    "chase_rand",
    "fnv",
    "fma",
    "vadd",
];

fn min_run(store: &mut Store<()>, f: &wasmtime::Func, n: i32) -> f64 {
    let mut out = [Val::I32(0)];
    f.call(&mut *store, &[Val::I32(n)], &mut out).unwrap(); // warm up (Cranelift compile happens at instantiate)
    let mut best = f64::MAX;
    for _ in 0..REPS {
        let t = Instant::now();
        f.call(&mut *store, &[Val::I32(n)], &mut out).unwrap();
        best = best.min(t.elapsed().as_nanos() as f64);
    }
    best
}

fn bench(label: &str, path: &str, memory64: bool) {
    let mut cfg = Config::new();
    cfg.wasm_memory64(memory64);
    let engine = Engine::new(&cfg).unwrap();
    let module = Module::from_file(&engine, path).unwrap();
    let mut store = Store::new(&engine, ());
    let inst = Instance::new(&mut store, &module, &[]).unwrap();
    for &name in KERNELS {
        let f = inst.get_func(&mut store, name).unwrap();
        let s = min_run(&mut store, &f, NS);
        let l = min_run(&mut store, &f, NL);
        println!("{label},{name},{:.4}", (l - s) / (NL - NS) as f64);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    bench("wasm32(wasmtime)", &args[1], false);
    bench("wasm64(wasmtime)", &args[2], true);
}
