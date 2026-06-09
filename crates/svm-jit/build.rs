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
    if arch == "x86_64" && (on_unix || on_windows) {
        println!("cargo:rustc-cfg=fiber_rt");
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
