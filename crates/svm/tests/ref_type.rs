//! `ValType::Ref` — the opaque 64-bit GC forward-compat reservation (GC.md §6).
//!
//! `ref` is reserved now so a future *precise* GC can name pointer-typed slots without a binary
//! format break. Today there is no instruction that *produces* a `ref` literal, but a `ref` value
//! can still flow as a param/result/block-arg, where it is operationally an opaque `i64`. These
//! tests pin that the type is genuinely wired end-to-end — text + binary round-trip and an
//! identity function that threads a `ref` through and returns it, bit-identically on interp + JIT.

use svm_encode::{decode_module, encode_module};
use svm_interp::{run, Value};
use svm_jit::{compile_and_run, JitOutcome};
use svm_text::{parse_module, print_module};
use svm_verify::verify_module;

/// `func (ref) -> (ref) { return v0 }` — the minimal program in which a `ref` value exists.
const IDENTITY: &str = "func (ref) -> (ref) {\n\
    block0(v0: ref):\n\
    \x20 return v0\n\
    }\n";

#[test]
fn ref_type_text_and_binary_round_trip() {
    let m = parse_module(IDENTITY).expect("parse ref module");
    verify_module(&m).expect("verify ref module");
    // Text is 1:1 with the IR (§3a): print → parse is the identity.
    assert_eq!(
        parse_module(&print_module(&m)),
        Ok(m.clone()),
        "text round-trip changed the ref module"
    );
    // Binary encode → decode is the identity (the new T_REF tag survives the wire).
    assert_eq!(
        decode_module(&encode_module(&m)),
        Ok(m.clone()),
        "binary round-trip changed the ref module"
    );
    // The type spells as `ref` in the text form.
    assert!(
        print_module(&m).contains("ref"),
        "the ref type must print as `ref`"
    );
}

#[test]
fn ref_value_threads_through_identically_on_interp_and_jit() {
    let m = parse_module(IDENTITY).expect("parse");
    verify_module(&m).expect("verify");

    // A `ref` is an opaque 64-bit word; thread an arbitrary bit-pattern through the identity fn.
    let bits: u64 = 0xDEAD_BEEF_0BAD_F00D;

    let mut fuel = 1_000_000u64;
    let interp = run(&m, 0, &[Value::Ref(bits)], &mut fuel).expect("interp run");
    assert_eq!(
        interp,
        vec![Value::Ref(bits)],
        "the interpreter must return the ref unchanged (opaque i64-width)"
    );

    match compile_and_run(&m, 0, &[bits as i64]).expect("jit compile/run") {
        JitOutcome::Returned(slots) => assert_eq!(
            slots,
            vec![bits as i64],
            "the JIT must return the ref slot unchanged (ref lowers as i64)"
        ),
        other => panic!("jit did not return: {other:?}"),
    }
}
