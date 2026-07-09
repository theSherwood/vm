//! Tiering-analysis unit tests (BROWSER.md § "wasm-JIT tier", slice 3). `analyze` classifies each
//! function as in-subset (the JIT emits it), an interp leaf (a cross-tier call runs it on the
//! bytecode engine), or neither — and decides whether a guest can run mixed-tier at all.

use svm_wasmjit::analyze;

fn m(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// A pure-integer, single-function compute kernel: in-subset, mixed_ok.
#[test]
fn pure_integer() {
    let a = analyze(&m(r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 3
  v2 = i64.mul v0 v1
  return v2
}"#));
    assert_eq!(a.in_subset, vec![true]);
    assert_eq!(a.interp_leaf, vec![false]);
    assert!(a.mixed_ok);
}

/// An integer caller with a **SIMD leaf** (integer signature — takes/returns i64 — but uses `v128`
/// internally, which is out of subset): the caller is in-subset, the leaf is interp-callable, the
/// guest is mixed_ok. This is the motivating mixed-tier shape. (Floats are now in-subset, so the
/// out-of-subset exemplar is `v128`.)
#[test]
fn integer_caller_simd_leaf() {
    let a = analyze(&m(r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = call 1 (v0)
  return v1
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i64x2.add v1 v1
  v3 = i64x2.extract_lane 0 v2
  return v3
}"#));
    assert_eq!(a.in_subset, vec![true, false]);
    assert_eq!(a.interp_leaf, vec![false, true]);
    assert!(a.reachable.iter().all(|&r| r));
    assert!(
        a.mixed_ok,
        "integer caller + memory-free SIMD leaf must run mixed-tier"
    );
}

/// Func 0 itself is SIMD → not in-subset → the JIT can't take the entry → not mixed_ok (the whole
/// guest stays on the interpreter).
#[test]
fn simd_entry_not_mixed() {
    let a = analyze(&m(r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i64x2.extract_lane 0 v1
  return v2
}"#));
    assert_eq!(a.in_subset, vec![false]);
    assert!(!a.mixed_ok);
}

/// A SIMD callee that TOUCHES MEMORY is not an interp leaf (its fresh window would diverge from the
/// shared one), so a guest calling it is not mixed_ok — it falls back to the full interpreter.
#[test]
fn memory_touching_leaf_blocks_mixed() {
    let a = analyze(&m(r#"
memory 16
func (i64) -> (i64) {
block0(v0: i64):
  v1 = call 1 (v0)
  return v1
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i64x2.extract_lane 0 v1
  v3 = i64.const 0
  i64.store v3 v2
  return v2
}"#));
    assert_eq!(a.in_subset, vec![true, false]);
    assert_eq!(
        a.interp_leaf,
        vec![false, false],
        "a memory-touching callee is not an interp leaf"
    );
    assert!(!a.mixed_ok);
}

/// A SIMD callee that itself CALLS another function is not a (true) leaf — transitive tiers are a
/// later refinement — so it blocks mixed_ok in this slice.
#[test]
fn non_leaf_simd_blocks_mixed() {
    let a = analyze(&m(r#"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = call 1 (v0)
  return v1
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64x2.splat v0
  v2 = i64x2.extract_lane 0 v1
  v3 = call 2 (v2)
  return v3
}
func (i64) -> (i64) {
block0(v0: i64):
  return v0
}"#));
    assert!(
        !a.interp_leaf[1],
        "a non-leaf SIMD function is not interp-callable yet"
    );
    assert!(!a.mixed_ok);
}

/// Concurrency anywhere reachable forces the whole guest to the interpreter (a JITted frame can't
/// unwind across a suspension).
#[test]
fn concurrency_not_mixed() {
    let a = analyze(&m(r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  v1 = i64.const 0
  return v1
}"#));
    assert!(
        !a.mixed_ok,
        "a guest that spawns/joins must not run on the JIT tier"
    );
}

/// An UNREACHABLE out-of-subset function doesn't block mixed_ok — only reachable functions matter.
#[test]
fn unreachable_out_of_subset_ok() {
    let a = analyze(&m(r#"
func (i64) -> (i64) {
block0(v0: i64):
  return v0
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = f64.convert_i64_s v0
  v2 = call 1 (v0)
  v3 = i64.trunc_sat_f64_s v1
  return v3
}"#));
    assert_eq!(a.reachable, vec![true, false], "func 1 is never called");
    assert!(
        a.mixed_ok,
        "an unreachable non-subset function is irrelevant"
    );
}
