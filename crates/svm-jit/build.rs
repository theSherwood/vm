fn main() {
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
