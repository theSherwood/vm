//! **`call.import` outlining** ([`svm_wasmjit::outline_cap_calls`], IMPORTS.md phase 3) — the same
//! transform as cap-call outlining, applied to an **executable manifest import**. A `call.import` is
//! a host-boundary op outside the emitter's compute subset, so an inline one keeps its whole
//! function on the interpreter; outlining hoists it into an integer-signature wrapper (a cross-tier
//! leaf) and the hot function emits. The import is dispatched on the interpreter tier through the
//! module's [`svm_interp::BoundImport`] bindings — no `resolve_imports` rewrite anywhere, so the
//! phase-1 "verified bytes are executed bytes" invariant holds on the wasm tier too.
//!
//! Mirrors `outline_capcalls.rs`: (1) outlining **flips emittability** for an import-bearing entry
//! (the manifest itself no longer blocks `emit_module`), and (2) **preserves semantics** (same
//! bindings, same result) — and the rewritten module still **verifies**.

use svm_interp::{bytecode, BoundImport, Host, Value};
use svm_wasmjit::{compile_module_reactor, outline_cap_calls};

// The entry (`f0`) is otherwise-emittable integer compute, but drives import 0 (a host-fn interface,
// type_id 13 op 0) inline: it computes `hostfn(10, 20) + 5`. The host-fn adds its two args, so the
// result is `10 + 20 + 5 = 35`. The handle arg `v0` is the vestigial `call.import` handle operand.
const SRC: &str = r#"
memory 16
import 0 "hostfn" (i64, i64) -> (i64)
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 10
  v2 = i64.const 20
  vr = call.import 0 v0 (v1, v2)
  v3 = i64.const 5
  vsum = i64.add vr v3
  return vsum
}
"#;

fn parse(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// Run `f0` on the bytecode interpreter with import 0 bound to a host-fn capability that adds its
/// two args. Services the module's `call.import` (inline before outlining, in the wrapper after)
/// identically either way.
fn run(m: &svm_ir::Module) -> Vec<Value> {
    let mut host = Host::new();
    let handle = host.grant_host_fn(Box::new(|op, args, _mem| {
        assert_eq!(op, 0, "the guest calls op 0");
        Ok(vec![args[0] + args[1]])
    }));
    host.set_import_bindings(vec![BoundImport::required(13, 0, handle)]);
    let mut r = bytecode::Reactor::open(m).expect("open reactor");
    let mut fuel = u64::MAX;
    r.call(0, &[Value::I32(handle)], &mut fuel, &mut host)
        .expect("run f0")
}

#[test]
fn outlining_flips_emittability_and_preserves_semantics() {
    let mut m = parse(SRC);

    // Before: the entry's inline `call.import` puts it outside the compute subset, so the reactor
    // emit fails (the whole guest would stay on the interpreter).
    assert!(
        compile_module_reactor(&m, 0, false).is_err(),
        "an inline call.import in the entry blocks emit",
    );
    let before = run(&m);
    assert_eq!(before, vec![Value::I64(35)], "10 + 20 (import) + 5");

    // Outline the call.import into a wrapper function.
    outline_cap_calls(&mut m);
    assert_eq!(m.funcs.len(), 2, "exactly one import wrapper is appended");
    assert_eq!(m.imports.len(), 1, "the manifest is untouched — no rewrite");
    svm_verify::verify_module(&m).expect("the outlined module verifies");

    // After: the entry is pure compute + a `Call`, so it emits — with the import manifest still
    // present (the relaxed emit_module rule); the wrapper stays cross-tier on the interpreter.
    let (_wasm, emitted) = compile_module_reactor(&m, 0, false).expect("emittable after outlining");
    assert_eq!(
        emitted,
        vec![true, false],
        "the entry emits to wasm; the call.import wrapper stays cross-tier",
    );

    // And the transform is semantics-preserving: same bindings, same result.
    let after = run(&m);
    assert_eq!(after, before, "outlining preserves interpreter semantics");
}
