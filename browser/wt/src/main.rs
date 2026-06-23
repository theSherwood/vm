//! wasm64 byte-feeding differential — the Wasmtime-embedding twin of `corpus.mjs`.
//!
//! `corpus.mjs` runs the 119-case corpus through the **wasm32** build in Node. wasm64 — the
//! production target — could only be validated by the embedded `--invoke` probes (the Wasmtime CLI
//! can't write the scratch buffers the alloc ABI needs). This harness closes that gap: it loads the
//! **wasm64** module through a real Wasmtime embedding (memory64), `svm_alloc`s + writes each corpus
//! module/stdin/window, calls the exports, reads results/streams/snapshots back, and compares to the
//! *same* `corpus.json` ground truth — so the differential is byte-identical on **both** targets.
//!
//! Usage (from `browser/`, after `cargo run --bin gencorpus`):
//!   cargo run --manifest-path wt/Cargo.toml --release -- \
//!     target/wasm64-unknown-unknown/release/svm_browser.wasm

use serde_json::Value as J;
use std::fs;
use wasmtime::*;

/// A loaded wasm module + the running embedding: the store, instance, and its linear memory.
struct Vm {
    store: Store<()>,
    inst: Instance,
    mem: Memory,
}

impl Vm {
    fn new(path: &str) -> Vm {
        // memory64 (incl. 64-bit tables) — the proposal Rust's wasm64 target emits.
        let mut cfg = Config::new();
        cfg.wasm_memory64(true);
        let engine = Engine::new(&cfg).expect("engine");
        let module = Module::from_file(&engine, path).expect("load wasm64 module");
        let mut store = Store::new(&engine, ());
        // The default build is import-free, so instantiate with no imports.
        let inst = Instance::new(&mut store, &module, &[]).expect("instantiate (no imports)");
        let mem = inst
            .get_memory(&mut store, "memory")
            .expect("module exports `memory`");
        Vm { store, inst, mem }
    }

    /// Call an export with explicit `Val` params, returning the single result widened to `i64`.
    fn invoke(&mut self, name: &str, params: &[Val]) -> i64 {
        let f = self
            .inst
            .get_func(&mut self.store, name)
            .unwrap_or_else(|| panic!("missing export `{name}`"));
        let mut out = [Val::I64(0)];
        f.call(&mut self.store, params, &mut out)
            .unwrap_or_else(|e| panic!("call `{name}` failed: {e}"));
        match out[0] {
            Val::I64(x) => x,
            Val::I32(x) => x as i64,
            ref v => panic!("`{name}` returned non-int {v:?}"),
        }
    }

    fn status(&mut self) -> i32 {
        self.invoke("svm_status", &[]) as i32
    }

    /// `svm_alloc(len)` then write `bytes`; returns `(ptr, len)`. Empty ⇒ `(0, 0)` (null).
    fn load(&mut self, bytes: &[u8]) -> (i64, i64) {
        if bytes.is_empty() {
            return (0, 0);
        }
        let ptr = self.invoke("svm_alloc", &[Val::I64(bytes.len() as i64)]);
        self.mem
            .write(&mut self.store, ptr as usize, bytes)
            .expect("write into linear memory");
        (ptr, bytes.len() as i64)
    }

    fn free(&mut self, ptr: i64, len: i64) {
        if ptr != 0 && len != 0 {
            // `svm_dealloc` returns void — call with a zero-length results slice.
            let f = self.inst.get_func(&mut self.store, "svm_dealloc").unwrap();
            f.call(&mut self.store, &[Val::I64(ptr), Val::I64(len)], &mut [])
                .expect("svm_dealloc");
        }
    }

    /// Read `len` bytes at `ptr` out of linear memory.
    fn read(&mut self, ptr: i64, len: i64) -> Vec<u8> {
        if len == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; len as usize];
        self.mem
            .read(&self.store, ptr as usize, &mut buf)
            .expect("read from linear memory");
        buf
    }

    /// Read a cdylib-managed `(ptr_fn, len_fn)` output buffer (stdout/stderr/snapshot).
    fn read_out(&mut self, ptr_fn: &str, len_fn: &str) -> Vec<u8> {
        let ptr = self.invoke(ptr_fn, &[]);
        let len = self.invoke(len_fn, &[]);
        self.read(ptr, len)
    }
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}
fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Tally + reporting.
struct Score {
    total: usize,
    fail: usize,
}
impl Score {
    fn check(&mut self, label: &str, ok: bool, detail: &str) {
        self.total += 1;
        if ok {
            println!("  {label}: match ({detail})");
        } else {
            self.fail += 1;
            println!("  {label}: FAIL ({detail})");
        }
    }
}

// JSON field helpers (i64s are carried as strings to preserve precision).
fn s(v: &J, k: &str) -> String {
    v.get(k).and_then(J::as_str).unwrap_or("").to_string()
}
fn i(v: &J, k: &str) -> i64 {
    match v.get(k) {
        Some(J::String(t)) => t.parse().unwrap(),
        Some(J::Number(n)) => n.as_i64().unwrap(),
        _ => 0,
    }
}
fn arr<'a>(c: &'a J, k: &str) -> &'a [J] {
    c.get(k).and_then(J::as_array).map(|v| &v[..]).unwrap_or(&[])
}
fn read_module(file: &str) -> Vec<u8> {
    fs::read(file).unwrap_or_else(|e| panic!("read {file}: {e}"))
}

fn main() {
    let wasm = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/wasm64-unknown-unknown/release/svm_browser.wasm".into());
    let corpus: J = serde_json::from_str(&fs::read_to_string("corpus.json").expect("corpus.json"))
        .expect("parse corpus.json");

    let mut vm = Vm::new(&wasm);
    let is64 = vm.invoke("svm_abi_is64", &[]) == 1;
    println!("module: {wasm}  ({})", if is64 { "wasm64" } else { "wasm32" });
    let mut sc = Score { total: 0, fail: 0 };

    // ---- compute-like: svm_run / svm_run0 (compute, fiber, tailcall, simd, reflect) ----------
    let run_compute_like = |vm: &mut Vm, sc: &mut Score, section: &str, export1: &str| {
        for m in arr(&corpus, section) {
            let bytes = read_module(&s(m, "file"));
            let nargs = i(m, "nargs");
            let mut bad = 0;
            let cases = arr(m, "cases");
            for case in cases {
                let arg = i(case, "arg");
                let want_status = i(case, "status") as i32;
                let want_value = i(case, "value");
                let (p, l) = vm.load(&bytes);
                let got = if export1 == "svm_run_reflect" {
                    vm.invoke(export1, &[Val::I64(p), Val::I64(l), Val::I64(arg)])
                } else if nargs == 0 {
                    vm.invoke("svm_run0", &[Val::I64(p), Val::I64(l)])
                } else {
                    vm.invoke("svm_run", &[Val::I64(p), Val::I64(l), Val::I64(arg)])
                };
                let st = vm.status();
                vm.free(p, l);
                let ok = st == want_status && (want_status != 0 || got == want_value);
                sc.total += 1;
                if !ok {
                    sc.fail += 1;
                    bad += 1;
                    println!(
                        "  {}({arg}): FAIL native {{s:{want_status},v:{want_value}}} wasm {{s:{st},v:{got}}}",
                        s(m, "name")
                    );
                }
            }
            println!("  {}: {}/{} match", s(m, "name"), cases.len() - bad, cases.len());
        }
    };
    println!("[compute / fiber / tailcall / simd]");
    run_compute_like(&mut vm, &mut sc, "compute", "svm_run");
    run_compute_like(&mut vm, &mut sc, "fiber", "svm_run");
    run_compute_like(&mut vm, &mut sc, "tailcall", "svm_run");
    run_compute_like(&mut vm, &mut sc, "simd", "svm_run");
    println!("[reflection]");
    run_compute_like(&mut vm, &mut sc, "reflect", "svm_run_reflect");

    // ---- powerbox: svm_run_pb + captured streams / exit -------------------------------------
    println!("[powerbox]");
    for c in arr(&corpus, "powerbox") {
        let (mp, ml) = vm.load(&read_module(&s(c, "file")));
        let (sp, sl) = vm.load(&unhex(&s(c, "stdin")));
        let got = vm.invoke("svm_run_pb", &[Val::I64(mp), Val::I64(ml), Val::I64(sp), Val::I64(sl)]);
        let st = vm.status();
        let out = hex(&vm.read_out("svm_stdout_ptr", "svm_stdout_len"));
        let err = hex(&vm.read_out("svm_stderr_ptr", "svm_stderr_len"));
        let exit = vm.invoke("svm_exit_code", &[]) as i32;
        vm.free(mp, ml);
        vm.free(sp, sl);
        let ws = i(c, "status") as i32;
        let ok = st == ws
            && (ws != 0 || got == i(c, "value"))
            && (ws != 5 || exit == i(c, "exit") as i32)
            && out == s(c, "stdout")
            && err == s(c, "stderr");
        sc.check(&s(c, "name"), ok, &format!("status {st}"));
    }

    // ---- capture + gc-roots: svm_run_capture + snapshot -------------------------------------
    let run_capture_like = |vm: &mut Vm, sc: &mut Score, section: &str| {
        for c in arr(&corpus, section) {
            let (mp, ml) = vm.load(&read_module(&s(c, "file")));
            let (ip, il) = vm.load(&unhex(&s(c, "init")));
            let got = vm.invoke(
                "svm_run_capture",
                &[Val::I64(mp), Val::I64(ml), Val::I64(ip), Val::I64(il), Val::I64(i(c, "arg"))],
            );
            let st = vm.status();
            let snap = hex(&vm.read_out("svm_snapshot_ptr", "svm_snapshot_len"));
            vm.free(mp, ml);
            vm.free(ip, il);
            let ws = i(c, "status") as i32;
            let ok = st == ws && (ws != 0 || (got == i(c, "value") && snap == s(c, "snapshot")));
            sc.check(&s(c, "name"), ok, &format!("snapshot {}B", snap.len() / 2));
        }
    };
    println!("[capture]");
    run_capture_like(&mut vm, &mut sc, "capture");
    println!("[gc.roots]");
    run_capture_like(&mut vm, &mut sc, "gcroots");

    // ---- durability: svm_run_durable + snapshot ---------------------------------------------
    println!("[durability]");
    for c in arr(&corpus, "durable") {
        let (mp, ml) = vm.load(&read_module(&s(c, "file")));
        let (ip, il) = vm.load(&unhex(&s(c, "init")));
        let got = vm.invoke(
            "svm_run_durable",
            &[Val::I64(mp), Val::I64(ml), Val::I64(ip), Val::I64(il), Val::I64(i(c, "clock"))],
        );
        let st = vm.status();
        let snap = hex(&vm.read_out("svm_snapshot_ptr", "svm_snapshot_len"));
        vm.free(mp, ml);
        vm.free(ip, il);
        let ws = i(c, "status") as i32;
        let ok = st == ws && (ws != 0 || (got == i(c, "value") && snap == s(c, "snapshot")));
        sc.check(&s(c, "name"), ok, &format!("value {got}, snapshot {}B", snap.len() / 2));
    }

    // ---- single-result cap powerboxes: nested / region / jit / dynlink -----------------------
    let run_single = |vm: &mut Vm, sc: &mut Score, section: &str, export: &str, extra_i32: Option<&str>| {
        for c in arr(&corpus, section) {
            let (mp, ml) = vm.load(&read_module(&s(c, "file")));
            let mut params = vec![Val::I64(mp), Val::I64(ml)];
            if let Some(k) = extra_i32 {
                params.push(Val::I32(i(c, k) as i32));
            }
            let got = vm.invoke(export, &params);
            let st = vm.status();
            vm.free(mp, ml);
            let ws = i(c, "status") as i32;
            let ok = st == ws && (ws != 0 || got == i(c, "value"));
            let label = match extra_i32 {
                Some(k) => format!("{}({k}={})", s(c, "name"), i(c, k)),
                None => s(c, "name"),
            };
            sc.check(&label, ok, &format!("status {st}, value {got}"));
        }
    };
    println!("[nested children]");
    run_single(&mut vm, &mut sc, "nested", "svm_run_nested", None);
    println!("[SharedRegion]");
    run_single(&mut vm, &mut sc, "region", "svm_run_region", None);
    println!("[guest-JIT]");
    run_single(&mut vm, &mut sc, "jit", "svm_run_jit", None);
    println!("[dynamic linking]");
    run_single(&mut vm, &mut sc, "dynlink", "svm_run_dynlink", Some("link"));

    // ---- scale: a 2 MiB stdin→stdout echo through svm_alloc ----------------------------------
    println!("[scale]");
    {
        const SIZE: usize = 2 << 20;
        let mut input = vec![0u8; SIZE];
        for (k, b) in input.iter_mut().enumerate() {
            *b = (k.wrapping_mul(2654435761) & 0xff) as u8;
        }
        let (mp, ml) = vm.load(&read_module("corpus/bigecho.svmbc"));
        let (sp, sl) = vm.load(&input);
        vm.invoke("svm_run_pb", &[Val::I64(mp), Val::I64(ml), Val::I64(sp), Val::I64(sl)]);
        let st = vm.status();
        let out = vm.read_out("svm_stdout_ptr", "svm_stdout_len");
        vm.free(mp, ml);
        vm.free(sp, sl);
        let ok = st == 0 && out == input;
        sc.check("bigecho", ok, &format!("{} MiB echoed, status {st}", SIZE >> 20));
    }

    println!(
        "\n{}/{} cases match native  {}",
        sc.total - sc.fail,
        sc.total,
        if sc.fail == 0 { "ALL MATCH" } else { "FAILED" }
    );
    std::process::exit(if sc.fail == 0 { 0 } else { 1 });
}
