fn main() {
    // The detect-and-kill trap shim (§4/§5) is unix-only; elsewhere the JIT falls back to a
    // plain heap window with no hardware guard (see `mem.rs`).
    #[cfg(unix)]
    {
        println!("cargo:rerun-if-changed=src/trap_shim.c");
        cc::Build::new()
            .file("src/trap_shim.c")
            .warnings(true)
            .compile("svm_trap_shim");
    }
}
