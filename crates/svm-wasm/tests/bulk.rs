//! Bulk-memory: **passive data segments + `memory.init` / `data.drop`** (the toolchain's
//! `__wasm_init_memory` shape). A passive segment's bytes are known at transpile time, so a
//! constant-offset `memory.init` unrolls into chunked const-stores of those bytes (no runtime
//! passive-data store, no IR change) — and `data.drop` is a no-op. Differential: interp == JIT, since
//! the lowering is plain `store`s both backends already run.

use svm_interp::Value;

/// Transpile WAT → IR, verify, run the export `entry` (no args) on interp + JIT, assert they agree,
/// return the single i32 result.
fn run(wat: &str, entry: &str) -> i64 {
    let wasm = wat::parse_str(wat).expect("assemble wat");
    let t = svm_wasm::transpile(&wasm).expect("transpile");
    svm_verify::verify_module(&t.module).expect("verify transpiled IR");
    let idx = t
        .exports
        .iter()
        .find(|(n, _)| n == entry)
        .unwrap_or_else(|| panic!("no export {entry}"))
        .1;
    let mut fuel = 100_000_000u64;
    let interp = svm_interp::run(&t.module, idx, &[], &mut fuel).expect("interp run");
    let jit = match svm_jit::compile_and_run(&t.module, idx, &[]).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    let iv = match interp[0] {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        other => panic!("unexpected interp value {other:?}"),
    };
    assert_eq!(iv as u32 as u64, jit[0] as u32 as u64, "interp != jit");
    iv
}

/// `memory.init` from a passive segment copies the segment's bytes into the window; `data.drop` then
/// runs (a no-op). Reading the four bytes back as a little-endian `i32` proves the copy landed.
#[test]
fn memory_init_copies_passive_segment() {
    let wat = r#"
      (module
        (memory 1)
        (data "\01\02\03\04")                              ;; passive (no offset)
        (func (export "f") (result i32)
          (memory.init 0 (i32.const 0) (i32.const 0) (i32.const 4))
          (data.drop 0)
          (i32.load (i32.const 0))))"#;
    assert_eq!(run(wat, "f"), 0x0403_0201, "0x04030201, little-endian");
}

/// A **partial** init — a non-zero source offset and a sub-segment length — copies exactly
/// `seg[src..src+len]` to `dest`. `src=1, len=2` of `10 20 30 40` → bytes `20 30` at `mem[0]`.
#[test]
fn memory_init_partial_range() {
    let wat = r#"
      (module
        (memory 1)
        (data "\10\20\30\40")
        (func (export "f") (result i32)
          (memory.init 0 (i32.const 0) (i32.const 1) (i32.const 2))
          (i32.load16_u (i32.const 0))))"#;
    assert_eq!(run(wat, "f"), 0x3020, "bytes 0x20,0x30 → 0x3020");
}

/// `memory.init` writes to a **runtime** destination address (only the source range must be constant).
/// Here `dest` is computed (`2 + 3`), so the bytes land at `mem[5]`.
#[test]
fn memory_init_to_runtime_dest() {
    let wat = r#"
      (module
        (memory 1)
        (data "\ab")
        (func (export "f") (result i32)
          (memory.init 0 (i32.add (i32.const 2) (i32.const 3)) (i32.const 0) (i32.const 1))
          (i32.load8_u (i32.const 5))))"#;
    assert_eq!(run(wat, "f"), 0xab);
}

/// Multiple data segments are indexed correctly — `memory.init 1` copies the **second** segment.
#[test]
fn memory_init_selects_the_right_segment() {
    let wat = r#"
      (module
        (memory 1)
        (data "\aa")            ;; segment 0
        (data "\bb")            ;; segment 1
        (func (export "f") (result i32)
          (memory.init 1 (i32.const 0) (i32.const 0) (i32.const 1))
          (i32.load8_u (i32.const 0))))"#;
    assert_eq!(run(wat, "f"), 0xbb);
}

/// An **active** segment's bytes are also reachable by `memory.init` (active segments enter the index
/// space too). The active copy places `cd` at `mem[8]` at instantiation; the init copies it to `mem[0]`.
#[test]
fn memory_init_from_an_active_segment() {
    let wat = r#"
      (module
        (memory 1)
        (data (i32.const 8) "\cd")
        (func (export "f") (result i32)
          (memory.init 0 (i32.const 0) (i32.const 0) (i32.const 1))
          (i32.load8_u (i32.const 0))))"#;
    assert_eq!(run(wat, "f"), 0xcd);
}

/// A **non-constant** `len` is fail-closed (clean `Unsupported`) — there is no runtime passive-data
/// store to read a dynamic count from.
#[test]
fn memory_init_dynamic_len_is_unsupported() {
    let wat = r#"
      (module
        (memory 1)
        (data "\01\02")
        (func (export "f") (param $n i32) (result i32)
          (memory.init 0 (i32.const 0) (i32.const 0) (local.get $n))
          (i32.load8_u (i32.const 0))))"#;
    let wasm = wat::parse_str(wat).expect("assemble wat");
    assert!(
        matches!(
            svm_wasm::transpile(&wasm),
            Err(svm_wasm::Error::Unsupported(_))
        ),
        "dynamic-len memory.init must be a clean Unsupported"
    );
}
