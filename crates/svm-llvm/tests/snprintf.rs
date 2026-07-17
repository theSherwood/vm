//! `snprintf` into a **stack** buffer across the formats Lua uses (`%d`/`%lld`/`%u`/`%x`/`%s`/`%p`/
//! `%.14g`), run through the powerbox on the tree-walker, bytecode, and JIT vs native. Regression for:
//! (1) the page-0 `FMT_BUF` write-protection fix — an snprintf program forces the powerbox layout so a
//! read-only global (the constant format string) does not share the writable format-scratch page; and
//! (2) `%p` support. `snprintf` reuses the whole printf format engine with output redirected into the
//! destination buffer.

use std::process::Command;
use svm_run::{Backend, Limits, Outcome, RunConfig};

/// Compile `src` (a `main` program returning a byte) to bitcode + a native exe, translate it, and run
/// it through the powerbox on `backend`; assert the result equals the native exit code. Skips cleanly
/// when the toolchain is unavailable.
fn check(name: &str, src: &str) {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("svm_snp_{}_{}.c", std::process::id(), name));
    let bc = dir.join(format!("svm_snp_{}_{}.ll", std::process::id(), name));
    let exe = dir.join(format!("svm_snp_{}_{}", std::process::id(), name));
    std::fs::write(&c, src).expect("write C source");
    let clang = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-S"])
        .arg(&c)
        .arg("-o")
        .arg(&bc)
        .status();
    match clang {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping {name} (clang unavailable)");
            return;
        }
    }
    match Command::new("cc")
        .arg(&c)
        .arg("-lm")
        .arg("-o")
        .arg(&exe)
        .status()
    {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("note: skipping {name} (cc unavailable)");
            return;
        }
    }
    let native = Command::new(&exe)
        .status()
        .expect("run native")
        .code()
        .unwrap() as u8;

    let t = svm_llvm::translate_ll_path(&bc).expect("translate bitcode");
    let inst = svm_run::instantiate(t.module).expect("instantiate");
    let config = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![],
        env: vec![],
    };
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let run = inst
            .run(backend, &config)
            .unwrap_or_else(|e| panic!("{name}: {backend:?} run failed: {e}"));
        let got = match run.outcome {
            Outcome::Returned(vs) => match vs.first() {
                Some(svm_interp::Value::I32(x)) => *x as u8,
                other => panic!("{name}: {backend:?} unexpected return {other:?}"),
            },
            Outcome::Exited(code) => code as u8,
        };
        assert_eq!(got, native, "{name}: {backend:?}={got} vs native={native}");
    }
}

#[test]
fn snprintf_stack_buffer_all_formats() {
    // A FNV-1a fold of each snprintf output (contents + length) into a return byte. The destination is
    // a stack array (the case that used to fault: the rodata format strings write-protected the page-0
    // FMT_BUF scratch). Mirrors Lua's number formats (%lld/%.14g) and error-message conversions
    // (%d/%s/%p).
    let src = "#include <stdio.h>\n\
        static unsigned long long fold(unsigned long long h, const char*b, int n){\n\
          for(int i=0;i<n && i<64;i++) h=(h^(unsigned char)b[i])*1099511628211ULL;\n\
          return (h^(unsigned)n)*1099511628211ULL; }\n\
        int main(void){\n\
          char b[64]; unsigned long long h=1469598103934665603ULL; int n;\n\
          n=snprintf(b,sizeof b,\"%d\",-12345); h=fold(h,b,n);\n\
          n=snprintf(b,sizeof b,\"%lld\",9876543210LL); h=fold(h,b,n);\n\
          n=snprintf(b,sizeof b,\"%u\",4000000000u); h=fold(h,b,n);\n\
          n=snprintf(b,sizeof b,\"%x\",0xdeadbeefu); h=fold(h,b,n);\n\
          n=snprintf(b,sizeof b,\"err %d: %s at %p\",7,\"index nil\",(void*)0x1234); h=fold(h,b,n);\n\
          n=snprintf(b,sizeof b,\"%.14g\",3.14159265358979); h=fold(h,b,n);\n\
          h^=h>>32; h^=h>>16; h^=h>>8;\n\
          return (int)(h & 0xff); }";
    check("snprintf_all_formats", src);
}
