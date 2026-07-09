// Native smoke runner for the on-ramp powerbox entry: decode a `.svmb` off `svm-llvm-translate`
// and run it through `svm_browser::onramp_exec`, printing status + captured stdout. Proves the
// on-ramp powerbox ABI (grant order, arity) with the *same* logic the wasm `svm_run_onramp` export
// runs — before the wasm/Chromium path. Usage: `cargo run --example run_onramp -- <file.svmb>`.
fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: run_onramp <file.svmb>");
    let bytes = std::fs::read(&path).expect("read .svmb");
    let m = svm_encode::decode_module(&bytes).expect("decode module");
    let out = svm_browser::onramp_exec(&m, b"");
    eprintln!(
        "status={} value={} exit={}",
        out.status, out.value, out.exit_code
    );
    print!("{}", String::from_utf8_lossy(&out.stdout));
    if !out.stderr.is_empty() {
        eprint!("[stderr] {}", String::from_utf8_lossy(&out.stderr));
    }
}
