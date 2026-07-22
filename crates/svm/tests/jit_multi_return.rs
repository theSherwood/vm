//! Multi-result function ABI on the JIT (target-uniform return-area / "sret").
//!
//! Cranelift's `Tail` calling convention returns values in registers, with a per-ABI budget — so a
//! function returning more results than fit (≈8+ on x86-64, fewer on aarch64) used to be *rejected*
//! by the JIT on one target while compiling on another (`Unsupported: Too many return values to fit
//! in registers, #9510`). That asymmetry — a valid module accepted on one supported target and not
//! another — is now gone: results beyond [`MAX_REG_RESULTS`] spill to a caller-provided memory
//! return-area pointer (like wasm engines do for multi-value), which is identical codegen on every
//! target. These differential tests (interp == JIT) exercise that path through every shape that
//! lays out the ABI: a plain multi-result return, mixed result types (the 8-byte-slot
//! encode/decode), and direct / indirect / tail calls to a many-result callee.

use svm_interp::{run, Value};
use svm_jit::{compile_and_run, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Encode an interpreter `Value` into its i64 calling-convention slot, matching the JIT's
/// `encode_slot` (i32/f32 zero-extended from 32 bits) so the two backends are comparable.
fn to_slot(v: &Value) -> i64 {
    match v {
        Value::I32(x) => *x as u32 as i64,
        Value::I64(x) => *x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
        other => panic!("unexpected result type in a multi-return test: {other:?}"),
    }
}

/// Run `src`'s function 0 on both backends and assert the (multi-)result agrees.
fn diff(src: &str) {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    let mut fuel = 1_000_000u64;
    let want: Vec<i64> = run(&m, 0, &[], &mut fuel)
        .unwrap_or_else(|t| panic!("interp trapped: {t:?}\n{src}"))
        .iter()
        .map(to_slot)
        .collect();
    let got = match compile_and_run(&m, 0, &[]).expect("jit compile/run") {
        JitOutcome::Returned(s) => s,
        other => panic!("jit did not return: {other:?}\n{src}"),
    };
    assert_eq!(want, got, "interp vs jit multi-result mismatch\n{src}");
}

/// **Nine results returned directly.** This exact shape (9 register returns) was rejected by the
/// x86-64 JIT with #9510 before the sret path existed; it must now compile and match the interpreter.
#[test]
fn nine_i64_results_returned_directly() {
    diff(
        "func () -> (i64, i64, i64, i64, i64, i64, i64, i64, i64) {\n\
         block 0 () {\n\
         \x20 v0 = i64.const 100\n\
         \x20 v1 = i64.const 101\n\
         \x20 v2 = i64.const 102\n\
         \x20 v3 = i64.const 103\n\
         \x20 v4 = i64.const 104\n\
         \x20 v5 = i64.const 105\n\
         \x20 v6 = i64.const 106\n\
         \x20 v7 = i64.const 107\n\
         \x20 v8 = i64.const 108\n\
         \x20 return v0 v1 v2 v3 v4 v5 v6 v7 v8\n\
           }\n\
         }\n",
    );
}

/// **Mixed result types past the register budget** — exercises the 8-byte-slot encode/decode on the
/// sret path (i32 results are zero-extended into / reduced out of their slots).
#[test]
fn mixed_i64_i32_results_via_sret() {
    diff(
        "func () -> (i64, i32, i64, i32, i64, i32) {\n\
         block 0 () {\n\
         \x20 v0 = i64.const 1000\n\
         \x20 v1 = i32.const 11\n\
         \x20 v2 = i64.const 2000\n\
         \x20 v3 = i32.const 22\n\
         \x20 v4 = i64.const 3000\n\
         \x20 v5 = i32.const 33\n\
         \x20 return v0 v1 v2 v3 v4 v5\n\
           }\n\
         }\n",
    );
}

/// Six i64 consts as a reusable callee body for the call tests.
const SIX_RESULT_CALLEE: &str = "func () -> (i64, i64, i64, i64, i64, i64) {\n\
     block 0 () {\n\
     \x20 v0 = i64.const 200\n\
     \x20 v1 = i64.const 201\n\
     \x20 v2 = i64.const 202\n\
     \x20 v3 = i64.const 203\n\
     \x20 v4 = i64.const 204\n\
     \x20 v5 = i64.const 205\n\
     \x20 return v0 v1 v2 v3 v4 v5\n\
       }\n\
     }\n";

/// **Direct call to a many-result callee**: the caller allocates the return-area, passes it, and
/// reads the six results back out of it.
#[test]
fn direct_call_to_many_result_fn() {
    diff(&format!(
        "func () -> (i64, i64, i64, i64, i64, i64) {{\n\
         block 0 () {{\n\
         \x20 v0, v1, v2, v3, v4, v5 = call 1()\n\
         \x20 return v0 v1 v2 v3 v4 v5\n\
           }}\n\
         }}\n{SIX_RESULT_CALLEE}"
    ));
}

/// **Indirect call to a many-result callee**: the sret decision is pinned by the call site's type,
/// so `call_indirect` allocates + threads the return-area exactly like a direct call.
#[test]
fn indirect_call_to_many_result_fn() {
    diff(&format!(
        "func () -> (i64, i64, i64, i64, i64, i64) {{\n\
         block 0 () {{\n\
         \x20 v0 = ref.func 1\n\
         \x20 v1, v2, v3, v4, v5, v6 = call_indirect () -> (i64, i64, i64, i64, i64, i64) v0()\n\
         \x20 return v1 v2 v3 v4 v5 v6\n\
           }}\n\
         }}\n{SIX_RESULT_CALLEE}"
    ));
}

/// **Tail call between many-result functions**: the callee shares the caller's result type, so the
/// caller forwards its own return-area pointer to the tail callee.
#[test]
fn tail_call_many_results() {
    diff(&format!(
        "func () -> (i64, i64, i64, i64, i64, i64) {{\n\
         block 0 () {{\n\
         \x20 return_call 1()\n\
           }}\n\
         }}\n{SIX_RESULT_CALLEE}"
    ));
}
