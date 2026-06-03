//! libFuzzer target for the escape-TCB invariants (`DESIGN.md` §2a / §18).
//!
//! Same contract as the stable `fuzz_smoke` test, run under coverage-guided
//! fuzzing: on arbitrary bytes, `decode` must fail-closed (never panic/OOM/hang),
//! `verify` must never panic, and any *verified* module must be safe to interpret
//! (bounded by fuel). A crash here is a candidate fail-open / escape bug.
//!
//! Run: `cargo +nightly fuzz run decode_verify`
#![no_main]

use libfuzzer_sys::fuzz_target;

use svm::default_args;
use svm_encode::decode_module;
use svm_interp::run;
use svm_verify::verify_module;

fuzz_target!(|data: &[u8]| {
    if let Ok(m) = decode_module(data) {
        if verify_module(&m).is_ok() {
            for (fi, f) in m.funcs.iter().enumerate() {
                let args = default_args(&f.params);
                let mut fuel = 10_000u64;
                let _ = run(&m, fi as u32, &args, &mut fuel);
            }
        }
    }
});
