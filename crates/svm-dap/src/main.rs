//! The `svm-dap` binary: an interpreter-backed Debug Adapter Protocol server speaking the
//! `Content-Length`-framed JSON wire protocol over stdin/stdout (DEBUGGING.md W5). Point a DAP
//! client (e.g. VS Code) at it to debug a guest on the reference interpreter.

fn main() -> std::io::Result<()> {
    svm_dap::run_stdio()
}
