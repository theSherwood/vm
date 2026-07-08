//! **Footprint — code size & memory (deployed-sandbox view).** Measures the *size* axis the speed
//! benchmarks miss, from the perspective of the **actual runtime**: this binary links only the SVM
//! runtime crates (`svm-ir`/`encode`/`interp`/`jit`) — **no LLVM anywhere**. The LLVM frontend is an
//! *AOT* tool (`svm-llvm-translate`) that produces SVM IR (`.svm`/`.svmb`); that IR is what travels
//! to the sandbox, and this probe consumes it just like a deployment would. So the RSS numbers here
//! are the real per-guest footprint, with none of the frontend's toolchain (clang/llvm-dis) present.
//!
//! For a module file (`.svmb` binary or `.svm` text) it reports:
//!   * **IR** bytes (`svm_encode::encode_module`) + instruction count — the shippable program;
//!   * **bytecode** op count (`Compiled::op_count`) — the threaded register-VM program size;
//!   * **JIT** native-code bytes (`CompiledModule::code_byte_count`) + IR→native expansion;
//!   * **peak process RSS** to build + hold each engine's artifact (re-exec'd per engine so the JIT's
//!     transient Cranelift compile working set is captured), minus a load-only baseline.
//!
//! Produce the input AOT (the frontend step, the only place LLVM tools are used), then probe:
//!   clang -O2 -emit-llvm -c bench/cross-engine/kernels.c -o /tmp/k.bc
//!   ( cd crates/svm-llvm && cargo run --release --bin svm-llvm-translate -- /tmp/k.bc -o /tmp/k.svmb )
//!   cargo run -p svm-run --release --example footprint -- /tmp/k.svmb

use std::ffi::c_void;
use std::path::Path;
use std::process::Command;

use svm_interp::bytecode;
use svm_ir::Module;

/// Peak (`VmHWM`) and current (`VmRSS`) resident set, in KiB, from `/proc/self/status`.
fn rss_kb() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |key: &str| {
        s.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
    };
    (get("VmHWM:"), get("VmRSS:"))
}

/// Load an SVM IR module from a `.svmb` (binary) or `.svm` (text) file — pure runtime, no libLLVM.
fn load(path: &str) -> Module {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    if path.ends_with(".svmb") {
        svm_encode::decode_module(&bytes).expect("decode .svmb")
    } else {
        svm_text::parse_module(&String::from_utf8(bytes).expect("utf8 .svm")).expect("parse .svm")
    }
}

/// Compile the whole module on the JIT (every function is lowered regardless of entry) and hold it.
fn jit_compile(m: &Module) -> (svm_jit::CompiledModule, usize) {
    let cm = svm_jit::CompiledModule::compile(
        m,
        0,
        svm_jit::INERT_CAP_THUNK,
        std::ptr::null_mut::<c_void>(),
        28, // reserved_log2: a virtual range only; doesn't affect code size or resident pages
        None,
        None,
        None,
        None,
        svm_jit::Quota::default(),
        0,
    )
    .expect("jit compile");
    let bytes = cm.code_byte_count();
    (cm, bytes)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // --- Child mode: build one engine's artifact, hold it across the RSS read, print "peak cur". ---
    if args.len() >= 4 && args[1] == "--rss" {
        let m = load(&args[3]);
        let _hold: Box<dyn std::any::Any> = match args[2].as_str() {
            "none" => Box::new(()), // load-only baseline (module decoded, nothing built)
            "tree-walk" => Box::new(m), // the module itself is the tree-walker's whole footprint
            "bytecode" => Box::new(bytecode::compile_module(&m.funcs).expect("bc compile")),
            "jit" => Box::new(jit_compile(&m).0),
            other => panic!("unknown engine {other}"),
        };
        let (peak, cur) = rss_kb();
        println!("{peak} {cur}");
        return;
    }

    let path = args.get(1).map(String::as_str).unwrap_or_else(|| {
        eprintln!(
            "usage: footprint <module.svmb|module.svm>\n  (produce the input AOT with \
             svm-llvm-translate; this probe links no libLLVM — see the file header)"
        );
        std::process::exit(2);
    });

    // --- Parent mode: size table + RSS table (re-exec per engine). All runtime-only, no libLLVM. ---
    let m = load(path);
    let ir_bytes = svm_encode::encode_module(&m).len();
    let ir_insts: usize = m
        .funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .map(|b| b.insts.len())
        .sum();
    let n_funcs = m.funcs.len();
    let bc_ops = bytecode::compile_module(&m.funcs)
        .expect("bc compile")
        .op_count();
    let (_cm, jit_bytes) = jit_compile(&m);

    let file = Path::new(path).file_name().unwrap().to_string_lossy();
    println!("module: {file} — {n_funcs} functions, {ir_insts} IR instructions (SVM IR off disk; no libLLVM)\n");
    println!(
        "{:<22} {:>12} {:>16}",
        "representation", "size", "vs IR bytes"
    );
    println!(
        "{:<22} {:>12} {:>16}",
        "IR (encoded)",
        format!("{ir_bytes} B"),
        "1.00x"
    );
    println!(
        "{:<22} {:>12} {:>16}",
        "bytecode (threaded)",
        format!("{bc_ops} ops"),
        format!("{:.2} ops/inst", bc_ops as f64 / ir_insts as f64)
    );
    println!(
        "{:<22} {:>12} {:>16}",
        "JIT (native code)",
        format!("{jit_bytes} B"),
        format!("{:.2}x", jit_bytes as f64 / ir_bytes as f64)
    );

    println!("\npeak process RSS to load IR + build + hold each engine's artifact (re-exec'd child, no libLLVM):");
    let exe = std::env::current_exe().unwrap();
    let measure = |engine: &str| -> (u64, u64) {
        let out = Command::new(&exe)
            .args(["--rss", engine, path])
            .output()
            .expect("re-exec");
        let s = String::from_utf8_lossy(&out.stdout);
        let mut it = s.split_whitespace();
        (
            it.next().unwrap_or("0").parse().unwrap_or(0),
            it.next().unwrap_or("0").parse().unwrap_or(0),
        )
    };
    let (base_peak, _) = measure("none");
    println!(
        "{:<14} {:>12} {:>16}",
        "engine", "peak(KiB)", "Δ vs load-only"
    );
    println!("{:<14} {:>12} {:>16}", "load-only", base_peak, "—");
    for engine in ["tree-walk", "bytecode", "jit"] {
        let (peak, _) = measure(engine);
        println!(
            "{engine:<14} {peak:>12} {:>15}K",
            peak as i64 - base_peak as i64
        );
    }
    println!(
        "\n(This binary links no libLLVM — the LLVM frontend is AOT-only (svm-llvm-translate → SVM IR);\n \
         the sandbox ships the IR, not the compiler. JIT native code is ~{:.1}x the IR byte size;\n \
         holding the module/bytecode adds ~0 RSS, while a JIT compile adds the Cranelift working set —\n \
         the real per-guest memory cost, with no frontend baggage in the figure.)",
        jit_bytes as f64 / ir_bytes as f64
    );
}
