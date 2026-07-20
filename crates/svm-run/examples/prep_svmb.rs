//! **Demo-artifact prep + boot-cost measurement.** Turn a raw translated `.svmb` (as emitted by
//! `svm-llvm-translate`, manifest imports intact) into the artifact a fast-loading demo ships:
//! verified, re-serialized. Along the way it times each phase of the load path — decode / verify /
//! bytecode-compile — so the one-time module-prep cost a pre-translated module still pays (vs
//! re-translating from bitcode at load) is measured, not guessed.
//!
//!   cargo run --release -p svm-run --example prep_svmb -- <in.svmb> [out.svmb]
//!
//! With `out.svmb`, writes the verified module there. The `.svmb` is the browser-loadable form (see
//! `BOOTSPEED.md`); the wasm module-prep tax is measured separately by `browser/bench_prep.mjs`.

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(input) = args.next() else {
        eprintln!("usage: prep_svmb <in.svmb> [out.svmb]");
        std::process::exit(2);
    };
    let output = args.next();

    let bytes = std::fs::read(&input).expect("read input .svmb");
    println!("module: {} ({} bytes)", input, bytes.len());

    let t = Instant::now();
    let module = svm_encode::decode_module(&bytes).expect("decode");
    println!(
        "  decode           {:>8.1?}  ({} funcs)",
        t.elapsed(),
        module.funcs.len()
    );

    // Phase 4: the runtime never rewrites — a powerbox entry keeps its manifest and the runtime
    // binds slots at instantiation, so there is no resolve phase left to measure.
    assert!(
        module.imports.is_empty() || svm_run::is_named_powerbox_entry(&module),
        "module declares imports but has no powerbox entry (IMPORTS.md phase 4)"
    );

    let t = Instant::now();
    svm_verify::verify_module(&module).expect("verify (fail-closed TCB)");
    println!(
        "  verify           {:>8.1?}  (mandatory — the trusted floor, never skippable)",
        t.elapsed()
    );

    let t = Instant::now();
    let compiled = svm_interp::bytecode::compile_module(&module.funcs);
    println!(
        "  bytecode compile {:>8.1?}  (interpreter cold cost, once at load; ok={})",
        t.elapsed(),
        compiled.is_some()
    );

    if let Some(out) = output {
        let t = Instant::now();
        let resolved = svm_encode::encode_module(&module);
        std::fs::write(&out, &resolved).expect("write resolved .svmb");
        println!(
            "wrote {} ({} bytes) in {:.1?}",
            out,
            resolved.len(),
            t.elapsed()
        );
    }
}
