fn main() {
    // `fiber_rt`: the JIT's fiber/thread/futex runtime (§12) is available wherever `svm-fiber`
    // provides a real stack switch. Keep this in lockstep with `svm_fiber::supported()` and the
    // `svm-fiber` module gates: x86-64 unix + x86-64 Windows today (aarch64 macOS next). Derived from
    // the *target* (not the host), so a cross-compile gates correctly. Registered unconditionally so
    // `#[cfg(fiber_rt)]` never trips the unexpected-cfg lint.
    println!("cargo:rustc-check-cfg=cfg(fiber_rt)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let on_unix = std::env::var_os("CARGO_CFG_UNIX").is_some();
    let on_windows = std::env::var_os("CARGO_CFG_WINDOWS").is_some();
    let fiber_rt =
        (on_unix && (arch == "x86_64" || arch == "aarch64")) || (on_windows && arch == "x86_64");
    if fiber_rt {
        println!("cargo:rustc-cfg=fiber_rt");
    }

    // The cross-platform trap-time backtrace capture (DEBUGGING.md §5 W3): the thread-local capture
    // state + frame-pointer walk + the explicit-trap helper, shared by the unix signal handler and the
    // windows VEH. Compiled wherever the trap-backtrace feature exists (unix + windows); other targets
    // (no-MMU / wasm) are unsupported (`compile_error!` in `mem.rs`) and don't reference these symbols.
    let on_windows = std::env::var_os("CARGO_CFG_WINDOWS").is_some();
    if std::env::var_os("CARGO_CFG_UNIX").is_some() || on_windows {
        println!("cargo:rerun-if-changed=src/trap_capture.c");
        let res = cc::Build::new()
            .file("src/trap_capture.c")
            .warnings(true)
            .try_compile("svm_trap_capture");
        if let Err(e) = res {
            // The windows-gnu *cross-check* (`cargo check/clippy --target …-windows-gnu` from a Linux
            // runner) has no mingw C compiler and doesn't link, so a missing cross-compiler there is
            // not fatal — the real windows build (MSVC on windows-latest) compiles this, and unix
            // always has `cc`. Any other failure (a genuine compile error) still aborts.
            let missing_tool =
                format!("{e:?}").contains("ToolNotFound") || e.to_string().contains("find tool");
            if on_windows && missing_tool {
                println!(
                    "cargo:warning=trap_capture.c: no C cross-compiler for this windows target \
                     (skipped — fine for a non-linking cross-check; MSVC compiles it natively)"
                );
            } else {
                panic!("trap_capture.c failed to compile: {e}");
            }
        }
    }

    // The detect-and-kill trap shim (§4/§5) is unix-only. Non-unix targets are unsupported and
    // refuse to build (`compile_error!` in `mem.rs`). Gate on the *target* family (`CARGO_CFG_UNIX`),
    // not the host: cross-compiling to Windows must hit that clean error, not a `cc`/mingw failure
    // here. (A build script's own `#[cfg(unix)]` reflects the host, so it can't be used for this.)
    if std::env::var_os("CARGO_CFG_UNIX").is_some() {
        println!("cargo:rerun-if-changed=src/trap_shim.c");
        cc::Build::new()
            .file("src/trap_shim.c")
            .warnings(true)
            .compile("svm_trap_shim");
    }
}
