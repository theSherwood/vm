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
    if std::env::var_os("CARGO_CFG_UNIX").is_some()
        || std::env::var_os("CARGO_CFG_WINDOWS").is_some()
    {
        println!("cargo:rerun-if-changed=src/trap_capture.c");
        cc::Build::new()
            .file("src/trap_capture.c")
            .warnings(true)
            .compile("svm_trap_capture");
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
