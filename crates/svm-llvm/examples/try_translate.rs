//! Scratch probe: translate a `.bc`/`.ll`, resolve capability imports, verify, run under the
//! powerbox, and (optionally) diff stdout against a native oracle binary. Used while closing
//! whole-program gaps (SQLite Phase A); not a test.
fn main() {
    let p = std::env::args()
        .nth(1)
        .expect("usage: try_translate <bc> [native-exe]");
    let native = std::env::args().nth(2);
    let t0 = std::time::Instant::now();
    // `.ll` → in-house textual reader; anything else → bitcode via `llvm-dis`.
    let is_ll = std::path::Path::new(&p)
        .extension()
        .is_some_and(|e| e == "ll");
    let translated = if is_ll {
        svm_llvm::translate_ll_path(&p)
    } else {
        svm_llvm::translate_bc_path(&p)
    };
    let t = match translated {
        Ok(t) => t,
        Err(e) => {
            println!("TRANSLATE ERR: {e:?}");
            std::process::exit(1);
        }
    };
    println!(
        "TRANSLATED in {:?}: {} funcs",
        t0.elapsed(),
        t.module.funcs.len()
    );
    // Phase 3: the manifest binds at instantiation — no rewrite.
    let module = t.module;
    if let Err(e) = svm_verify::verify_module(&module) {
        println!("VERIFY ERR: {e:?}");
        std::process::exit(1);
    }
    println!("VERIFIED");
    let t1 = std::time::Instant::now();
    let run = match svm_run::run_powerbox(&module, b"") {
        Ok(r) => r,
        Err(e) => {
            println!("RUN ERR: {e}");
            std::process::exit(1);
        }
    };
    println!("RAN in {:?}: outcome {:?}", t1.elapsed(), run.outcome);
    if let Some(exe) = native {
        let out = std::process::Command::new(&exe)
            .output()
            .expect("run native");
        if run.stdout == out.stdout {
            println!("STDOUT MATCHES NATIVE ({} bytes)", out.stdout.len());
        } else {
            println!(
                "STDOUT MISMATCH: svm {} bytes vs native {} bytes",
                run.stdout.len(),
                out.stdout.len()
            );
            let sv = String::from_utf8_lossy(&run.stdout);
            let nv = String::from_utf8_lossy(&out.stdout);
            for (i, (a, b)) in sv.lines().zip(nv.lines()).enumerate() {
                if a != b {
                    println!("line {}: svm    {a:?}", i + 1);
                    println!("line {}: native {b:?}", i + 1);
                    break;
                }
            }
            std::process::exit(2);
        }
    } else {
        println!("--- stdout ---\n{}", String::from_utf8_lossy(&run.stdout));
    }
}
