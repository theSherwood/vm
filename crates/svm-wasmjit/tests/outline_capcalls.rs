//! **Cap-call outlining** ([`svm_wasmjit::outline_cap_calls`]) — the transform that lets a reactor
//! whose hot `tick` interleaves compute with an inline `cap.call` (a `display.present` / `keyboard.poll`
//! once per frame) emit to wasm without a source change. A `cap.call` is outside the emitter's compute
//! subset, and emittability is per whole function, so one inline cap call keeps the whole function
//! (its hot loop included) on the interpreter. Outlining hoists each `cap.call` into an integer-
//! signature wrapper — a cross-tier leaf — so the hot function becomes pure compute + a `Call` and emits.
//!
//! This proves the transform's two contracts at the interpreter level: it (1) **flips emittability**
//! (a `cap.call`-bearing entry goes from `Unsupported` to emittable, with the wrapper cross-tier), and
//! (2) **preserves semantics** (running the module before vs after outlining, with the same host
//! servicing the cap, yields the identical result) — and that the rewritten module still **verifies**.
//! (The emitted↔interpreter cross-tier bridge is proven by `cross_tier.rs`; the full end-to-end with a
//! real `display` cap on the f64 Mandelbrot guest is proven by the browser JIT-reactor test.)

use svm_interp::{bytecode, Host, Value};
use svm_wasmjit::{compile_module_reactor, outline_cap_calls};

// The entry (`f0`) is otherwise-emittable integer compute, but makes one inline `cap.call` to a
// host-fn capability (`iface::HOST_FN` = type_id 13, op 0) whose handle arrives as the arg: it computes
// `host_fn(10, 20) + 5`. The host-fn adds its two args, so the result is `10 + 20 + 5 = 35`.
const SRC: &str = r#"
memory 16
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 10
  v2 = i64.const 20
  vr = cap.call 13 0 (i64, i64) -> (i64) v0(v1, v2)
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

/// Run `f0` on the bytecode interpreter with a host-fn capability that adds its two args, passing the
/// granted handle as the arg. Services the module's `cap.call` (inline before outlining, in the wrapper
/// after) identically either way.
fn run(m: &svm_ir::Module) -> Vec<Value> {
    let mut host = Host::new();
    let handle = host.grant_host_fn(Box::new(|op, args, _mem| {
        assert_eq!(op, 0, "the guest calls op 0");
        Ok(vec![args[0] + args[1]])
    }));
    let mut r = bytecode::Reactor::open(m).expect("open reactor");
    let mut fuel = u64::MAX;
    r.call(0, &[Value::I32(handle)], &mut fuel, &mut host)
        .expect("run f0")
}

#[test]
fn outlining_flips_emittability_and_preserves_semantics() {
    let mut m = parse(SRC);

    // Before: the entry's inline `cap.call` puts it outside the compute subset, so the reactor emit
    // fails (the whole guest, hot loop included, would stay on the interpreter).
    assert!(
        compile_module_reactor(&m, 0, false).is_err(),
        "an inline cap.call in the entry blocks emit",
    );
    let before = run(&m);
    assert_eq!(before, vec![Value::I64(35)], "10 + 20 (host_fn) + 5");

    // Outline the cap.call into a wrapper function.
    outline_cap_calls(&mut m);
    assert_eq!(m.funcs.len(), 2, "exactly one cap-call wrapper is appended");
    // The rewritten module must still be well-formed (the wrapper's signature, value numbering, and
    // the 1:1 call-site rewrite all check out).
    svm_verify::verify_module(&m).expect("the outlined module verifies");

    // After: the entry is now pure compute + a `Call`, so it emits; the wrapper (which holds the
    // cap.call, an all-integer signature) is a cross-tier leaf.
    let (_wasm, emitted) =
        compile_module_reactor(&m, 0, false).expect("emittable after outlining");
    assert_eq!(
        emitted,
        vec![true, false],
        "the entry emits to wasm; the cap.call wrapper stays cross-tier",
    );

    // And the transform is semantics-preserving: same host, same result.
    let after = run(&m);
    assert_eq!(after, before, "outlining preserves interpreter semantics");
}
