//! `svm-llvm-translate` — translate a legalized LLVM bitcode file (`*.bc`) to an SVM-IR module.
//!
//! ```text
//! svm-llvm-translate <input.bc> -o <out> [--emit-syms <file>] [--binary]
//! ```
//!
//! This is the **separate-artifact** on-ramp (the scriptable companion to the [`svm_llvm`] library):
//! a frontend like JACL compiles its runtime once to bitcode, translates it here to a reusable
//! `.svm`/`.svmb` module, and emits a `.syms` **export sidecar** (one `name idx` line per exported
//! function). A program module then resolves a `call.import` of those names by pairing the module
//! with its sidecar into a [`svm_ir::LinkUnit`] and running [`svm_ir::link`] — compile the runtime
//! once, link many programs against it.
//!
//! Output format: text (`svm_text::print_module`) by default, binary (`svm_encode::encode_module`)
//! when `-o` ends in `.svmb` or `--binary` is given.

use std::path::Path;
use std::{env, fs, process};

fn main() {
    if let Err(e) = try_main() {
        eprintln!("svm-llvm-translate: {e}");
        process::exit(1);
    }
}

fn try_main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: svm-llvm-translate <input.bc> -o <out> [--emit-syms <file>] [--binary]\n\
             \n  Translates legalized LLVM-18 bitcode to an SVM-IR module written to <out>:\n\
             \n    text (.svm) by default, binary (.svmb) when -o ends in .svmb or --binary.\n\
             \n  --emit-syms <file> writes the export map (one `name idx` line per exported\n\
             \n  function) so a program can link against the module via svm_ir::link."
        );
        return Err("no input file".into());
    }

    let mut input: Option<String> = None;
    let mut out: Option<String> = None;
    let mut syms: Option<String> = None;
    let mut binary = false;
    let mut stub_externs = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" | "--output" => out = Some(it.next().ok_or("-o needs a file argument")?.clone()),
            "--emit-syms" => {
                syms = Some(
                    it.next()
                        .ok_or("--emit-syms needs a file argument")?
                        .clone(),
                )
            }
            "--binary" => binary = true,
            // Lower undefined externals to trap-if-called stubs instead of failing translation — for a
            // large-program bring-up (Postgres) where most externals are dead on the exercised path.
            "--stub-externs" => stub_externs = true,
            _ if a.starts_with('-') => return Err(format!("unknown flag `{a}`")),
            _ => {
                if input.replace(a.clone()).is_some() {
                    return Err("more than one input file given".into());
                }
            }
        }
    }
    let input = input.ok_or("no input file")?;
    let out = out.ok_or("no output file (-o <out>)")?;
    // Binary if asked explicitly or the output names a `.svmb` file; text otherwise.
    let binary = binary || Path::new(&out).extension().is_some_and(|e| e == "svmb");

    // Translate the bitcode. `Error` is `Debug`-only (no `Display`), so render it that way.
    let opts = svm_llvm::TranslateOptions {
        stub_unresolved_externs: stub_externs,
    };
    let translated = svm_llvm::translate_bc_path_with_options(&input, opts)
        .map_err(|e| format!("translate `{input}`: {e:?}"))?;

    let module_bytes = if binary {
        svm_encode::encode_module(&translated.module)
    } else {
        svm_text::print_module(&translated.module).into_bytes()
    };
    fs::write(&out, &module_bytes).map_err(|e| format!("write `{out}`: {e}"))?;

    if let Some(syms) = syms {
        // One `name idx` line per exported function — the index is the function's slot in
        // `module.funcs`, which a linker pairs with the module to populate `LinkUnit.exports`.
        let mut s = String::new();
        for (name, idx) in &translated.exports {
            s.push_str(&format!("{name} {idx}\n"));
        }
        fs::write(&syms, s).map_err(|e| format!("write `{syms}`: {e}"))?;
    }

    Ok(())
}
