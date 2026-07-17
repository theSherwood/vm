//! **`cap.self.resolve` outlining** ([`svm_wasmjit::outline_cap_calls`] also hoists
//! [`svm_ir::Inst::CapSelfResolve`]) — the transform that lets the on-ramp `_start` synth emit. The
//! powerbox entry is otherwise pure compute + stores, but resolves each granted capability **by name**
//! (`cap.self.resolve`, a host-boundary reflection op outside the compute subset), so one such call
//! kept func 0 — and, under the whole-module `call_indirect` rule, the *entire* guest — off the wasm
//! tier. Outlining each `cap.self.resolve` into an integer-signature cross-tier wrapper makes the entry
//! pure compute + a `Call`, so it emits (see `outline_capcalls.rs` for the `cap.call` sibling).
//!
//! Proves the same two contracts at the interpreter level: outlining (1) **flips emittability** of a
//! `cap.self.resolve`-bearing entry, and (2) **preserves semantics** (same host, same resolved handle).

use svm_interp::{bytecode, Host, Value};
use svm_wasmjit::{compile_module_reactor, outline_cap_calls};

// The entry resolves the capability name "exit" — four bytes in a data segment at window offset 0 — to
// the handle it was granted under, and returns it. Pure compute apart from the one `cap.self.resolve`.
const SRC: &str = r#"
memory 16
data 0 "exit"
func () -> (i32) {
block0():
  v0 = i64.const 0
  v1 = i64.const 4
  v2 = cap.self.resolve v0 v1
  return v2
}
"#;

fn parse(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// Run `f0` on the bytecode interpreter with an `exit` capability registered under that name, so
/// `cap.self.resolve("exit")` returns its handle. Services the resolve identically whether it is inline
/// (before outlining) or in the wrapper (after).
fn run(m: &svm_ir::Module) -> Vec<Value> {
    let mut host = Host::new();
    let handle = host.grant_exit();
    host.register_cap_name("exit", handle);
    let mut r = bytecode::Reactor::open(m).expect("open reactor");
    let mut fuel = u64::MAX;
    let out = r.call(0, &[], &mut fuel, &mut host).expect("run f0");
    assert_eq!(
        out,
        vec![Value::I32(handle)],
        "resolve returns the granted handle"
    );
    out
}

#[test]
fn outlining_capself_resolve_flips_emittability_and_preserves_semantics() {
    let mut m = parse(SRC);

    // Before: the entry's inline `cap.self.resolve` puts it outside the compute subset, so the reactor
    // emit fails (the entry must be in-subset).
    assert!(
        compile_module_reactor(&m, 0, false).is_err(),
        "an inline cap.self.resolve in the entry blocks emit",
    );
    let before = run(&m);

    // Outline the `cap.self.resolve` into a wrapper function.
    outline_cap_calls(&mut m);
    assert_eq!(
        m.funcs.len(),
        2,
        "exactly one cap.self.resolve wrapper is appended"
    );
    svm_verify::verify_module(&m).expect("the outlined module verifies");

    // After: the entry is pure compute + a `Call`, so it emits; the wrapper (integer signature
    // `(i64, i64) -> i32`) is cross-tier.
    let (_wasm, emitted) = compile_module_reactor(&m, 0, false).expect("emittable after outlining");
    assert_eq!(
        emitted,
        vec![true, false],
        "the entry emits to wasm; the cap.self.resolve wrapper stays cross-tier",
    );

    // Semantics-preserving: same host, same resolved handle.
    let after = run(&m);
    assert_eq!(after, before, "outlining preserves interpreter semantics");
}
