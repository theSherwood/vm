//! **Footprint — code size & memory.** Every other driver measures *speed*; this measures *size*,
//! the other axis that matters for a capability VM meant to host many sandboxed guests. For the shared
//! `bench/cross-engine/kernels.c` (via the real LLVM frontend) it reports, per representation:
//!
//!  * **IR** — the serialized SVM IR (`svm_encode::encode_module`) in bytes, plus instruction count:
//!    the shippable/stored program form, and the baseline the other tiers expand from.
//!  * **bytecode** — the tree-/bytecode engines' threaded register-VM program (`bytecode::compile_module`),
//!    measured in **ops** (it's a `Vec<Op>`, not a byte stream).
//!  * **JIT** — the finalized native machine code (`CompiledModule::code_byte_count`), in bytes, plus
//!    the **IR→native expansion** factor.
//!
//! It then re-execs itself once per engine to capture **peak process RSS** (the high-water memory to
//! translate + build + hold that engine's artifact), minus a translate-only baseline — so the JIT's
//! transient Cranelift compile memory shows up, not just the retained code. Run:
//!   cd crates/svm-llvm && cargo run --release --example footprint

use std::path::PathBuf;
use std::process::Command;

use svm_interp::bytecode;

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

fn kernels_bc() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let kernels_c = root.join("bench/cross-engine/kernels.c");
    let bc = std::env::temp_dir().join(format!("svm_fp_{}.bc", std::process::id()));
    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-c"])
        .arg(&kernels_c)
        .arg("-o")
        .arg(&bc)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "clang -emit-llvm failed (need clang + libLLVM-18)");
    bc
}

/// Compile the entry on the JIT and hold it; returns the held module so the caller controls its
/// lifetime (for RSS), plus its finalized code-byte count.
fn jit_compile(m: &svm_ir::Module, e: u32) -> (svm_jit::CompiledModule, usize) {
    let cm = svm_jit::CompiledModule::compile(
        m,
        e,
        svm_jit::INERT_CAP_THUNK,
        std::ptr::null_mut(),
        28, // reserved_log2: a virtual range only; does not affect code size or resident pages
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

    // --- Child mode: build one engine's artifact, hold it, print "peak cur" RSS in KiB, exit. ---
    if args.len() >= 3 && args[1] == "--rss" {
        let bc = std::env::temp_dir().join(&args[3]);
        let t = svm_llvm::translate_bc_path(&bc).expect("translate");
        let e = t
            .exports
            .iter()
            .find(|(n, _)| n == "fnv")
            .map_or(0, |x| x.1);
        // Hold the artifact live across the RSS read so retained memory is counted.
        let _hold: Box<dyn std::any::Any> = match args[2].as_str() {
            "none" => Box::new(()),     // translate-only baseline
            "tree-walk" => Box::new(t), // the module itself is the tree-walker's whole footprint
            "bytecode" => Box::new(bytecode::compile_module(&t.module.funcs).expect("bc compile")),
            "jit" => Box::new(jit_compile(&t.module, e).0),
            other => panic!("unknown engine {other}"),
        };
        let (peak, cur) = rss_kb();
        println!("{peak} {cur}");
        return;
    }

    // --- Parent mode: size table (in-process) + RSS table (re-exec per engine). ---
    let bc = kernels_bc();
    let bc_name = bc.file_name().unwrap().to_str().unwrap().to_string();
    let t = svm_llvm::translate_bc_path(&bc).expect("translate kernels.c");

    let ir_bytes = svm_encode::encode_module(&t.module).len();
    let ir_insts: usize = t
        .module
        .funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .map(|b| b.insts.len())
        .sum();
    let n_funcs = t.module.funcs.len();
    let bc_ops = bytecode::compile_module(&t.module.funcs)
        .expect("bc compile")
        .op_count();
    // JIT code bytes: compile the whole module reachable from a representative entry.
    let e = t
        .exports
        .iter()
        .find(|(n, _)| n == "fnv")
        .map_or(0, |x| x.1);
    let (_cm, jit_bytes) = jit_compile(&t.module, e);

    println!("module: {n_funcs} functions, {ir_insts} IR instructions (kernels.c via the LLVM frontend)\n");
    println!(
        "{:<22} {:>12} {:>14}",
        "representation", "size", "vs IR bytes"
    );
    println!(
        "{:<22} {:>12} {:>14}",
        "IR (encoded)",
        format!("{ir_bytes} B"),
        "1.00x"
    );
    println!(
        "{:<22} {:>12} {:>14}",
        "bytecode (threaded)",
        format!("{bc_ops} ops"),
        format!("{:.2} ops/inst", bc_ops as f64 / ir_insts as f64)
    );
    println!(
        "{:<22} {:>12} {:>14}",
        "JIT (native code)",
        format!("{jit_bytes} B"),
        format!("{:.2}x", jit_bytes as f64 / ir_bytes as f64)
    );

    // RSS: re-exec per engine; subtract the translate-only ("none") baseline.
    println!(
        "\npeak process RSS to translate + build + hold each engine's artifact (re-exec'd child):"
    );
    let exe = std::env::current_exe().unwrap();
    let measure = |engine: &str| -> (u64, u64) {
        let out = Command::new(&exe)
            .args(["--rss", engine, &bc_name])
            .output()
            .expect("re-exec");
        let s = String::from_utf8_lossy(&out.stdout);
        let mut it = s.split_whitespace();
        (
            it.next().unwrap_or("0").parse().unwrap_or(0),
            it.next().unwrap_or("0").parse().unwrap_or(0),
        )
    };
    let (base_peak, base_cur) = measure("none");
    println!(
        "{:<14} {:>12} {:>14} {:>16}",
        "engine", "peak(KiB)", "retained(KiB)", "Δpeak vs base"
    );
    println!(
        "{:<14} {:>12} {:>14} {:>16}",
        "translate-only", base_peak, base_cur, "—"
    );
    for engine in ["tree-walk", "bytecode", "jit"] {
        let (peak, cur) = measure(engine);
        println!(
            "{engine:<14} {peak:>12} {cur:>14} {:>15}K",
            peak as i64 - base_peak as i64
        );
    }
    println!(
        "\n(IR is the shippable program; bytecode is a threaded register-VM (op count, not bytes);\n \
         JIT native code is ~{:.1}x the IR byte size. RSS deltas are dominated by Cranelift's compile\n \
         working set for the JIT — the transient cost the steady-state and code-size numbers don't show.)",
        jit_bytes as f64 / ir_bytes as f64
    );
    let _ = std::fs::remove_file(&bc);
}
