//! Soundness harness for the bytecode engine's **§GC `gc.roots`** (INTERP_PERF.md Slice 1c-7).
//!
//! Unlike every other bytecode harness, `gc.roots` is **not** checked bit-identical to the
//! tree-walker: the op is conservative root enumeration, and the backends legitimately
//! over-approximate differently (GC.md §3.2 — the JIT scans raw native control-stack words, the
//! tree-walker scans per-block `frame.vals`, and the bytecode engine scans each activation's whole
//! register window, which also retains dead values from other blocks of the same function). The
//! correctness criterion is therefore **soundness**:
//!   1. the bytecode engine reports *every* root the tree-walker does (`tw ⊆ bc`) — it never misses a
//!      live root, so a guest GC built on it can't free a reachable object;
//!   2. every reported word is in-window (`lo ≤ w < hi`) — no host address leaks past the range
//!      filter (the mask is constrained to top-byte-strip only);
//!   3. the planted genuine roots are all found, and `total` equals the set size.
//!
//! The two backends agree on the *format* (ascending, deduplicated, first `cap` written, total
//! returned). Memory is read back via `compile_and_run_capture` / `run_capture_reserved`.

use std::collections::BTreeSet;
use svm_interp::{bytecode, run_capture_reserved, Trap, Value};
use svm_ir::DEFAULT_RESERVED_LOG2;
use svm_text::parse_module;

/// Parse the roots an engine wrote: `total` is its `i64` result; the buffer at window offset `buf`
/// holds `total` little-endian `i64` words (the first `cap`, here always ≥ total).
fn roots_of(res: &Result<Vec<Value>, Trap>, snap: &[u8], buf: usize) -> (i64, BTreeSet<u64>) {
    let total = match res {
        Ok(v) => match v.first() {
            Some(Value::I64(t)) => *t,
            other => panic!("expected i64 total, got {other:?}"),
        },
        Err(e) => panic!("unexpected trap: {e:?}"),
    };
    let mut set = BTreeSet::new();
    for i in 0..total as usize {
        let off = buf + i * 8;
        set.insert(u64::from_le_bytes(snap[off..off + 8].try_into().unwrap()));
    }
    (total, set)
}

/// Run `src` on both engines (buffer at window offset 0), assert soundness: `tw ⊆ bc`, every bytecode
/// root is in `[lo, hi)`, and `expected ⊆ bc`.
fn check(src: &str, lo: u64, hi: u64, expected: &[u64]) {
    let m = parse_module(src).expect("parse");
    let init = vec![0u8; 4096];

    let mut f_tw = 1_000_000u64;
    let (tw_res, tw_snap) =
        run_capture_reserved(&m, 0, &[], &mut f_tw, &init, DEFAULT_RESERVED_LOG2);
    let mut f_bc = 1_000_000u64;
    let (bc_res, bc_snap) = bytecode::compile_and_run_capture(&m, 0, &[], &mut f_bc, &init)
        .expect("bytecode engine must support gc.roots (Slice 1c-7)");

    let (tw_total, tw_set) = roots_of(&tw_res, &tw_snap, 0);
    let (bc_total, bc_set) = roots_of(&bc_res, &bc_snap, 0);

    assert_eq!(
        tw_total as usize,
        tw_set.len(),
        "tree-walker total != set size"
    );
    assert_eq!(
        bc_total as usize,
        bc_set.len(),
        "bytecode total != set size"
    );
    // Soundness: the bytecode engine reports every root the tree-walker found (never misses one).
    assert!(
        tw_set.is_subset(&bc_set),
        "bytecode missed a root the tree-walker found\n  tw={tw_set:?}\n  bc={bc_set:?}\n{src}"
    );
    // No host leak: every reported word is an in-window guest value.
    for &w in &bc_set {
        assert!(
            (lo..hi).contains(&w),
            "bytecode reported out-of-window root {w:#x} (range [{lo:#x},{hi:#x}))\n{src}"
        );
    }
    // The planted genuine roots are all found.
    for &e in expected {
        assert!(
            bc_set.contains(&e),
            "expected genuine root {e:#x} missing from bytecode set {bc_set:?}\n{src}"
        );
    }
}

/// Baseline: a single block holds several in-range constants (one duplicated, one out of range). Both
/// backends see exactly the same `frame.vals`, so the sets are equal — `{4096, 5000}`.
const BASELINE: &str = r#"memory 16
func () -> (i64) {
block0():
  va = i64.const 4096
  vb = i64.const 5000
  vc = i64.const 5000
  vd = i64.const 9000
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 0
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
}
"#;

#[test]
fn baseline_caller_frame_roots() {
    check(BASELINE, 4096, 8192, &[4096, 5000]);
}

/// Strict superset: a value computed in `block0` is dead after the branch (not passed to `block1`).
/// The tree-walker resets `frame.vals` on the edge, so it does **not** report `5000`; the bytecode
/// engine's register window still holds it, so it does — a sound conservative over-approximation
/// (`tw = {4096} ⊊ bc = {4096, 5000}`). This is the case that proves the criterion is soundness, not
/// equality.
const CROSS_BLOCK_DEAD: &str = r#"memory 16
func () -> (i64) {
block0():
  vdead = i64.const 5000
  br block1()
block1():
  va = i64.const 4096
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 0
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
}
"#;

#[test]
fn cross_block_dead_value_is_sound_superset() {
    check(CROSS_BLOCK_DEAD, 4096, 8192, &[4096]);
}

/// Tagged pointers: each candidate carries a tag in the top byte; `mask` strips it so the bare offset
/// is range-tested and emitted. Both backends apply the same mask, so both recover `5000`.
const TAGGED: &str = r#"memory 16
func () -> (i64) {
block0():
  va = i64.const 9151314442816852872
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const 72057594037927935
  vbuf = i64.const 0
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
}
"#;

#[test]
fn tagged_pointer_mask_recovers_offset() {
    // va = (0x7F << 56) | 5000; mask = 0x00FF_FFFF_FFFF_FFFF → bare 5000 in range.
    check(TAGGED, 4096, 8192, &[5000]);
}

/// A root held in the **caller's** frame (live across a call — used after) is enumerated when the
/// callee runs `gc.roots` (the op is call-clobbering; the caller's live values are scannable on both
/// backends). The callee also contributes its own in-range constant.
const CALLER_FRAME: &str = r#"memory 16
func () -> (i64) {
block0():
  vroot = i64.const 5000
  vt = call 1()
  vsum = i64.add vt vroot
  return vsum
}
func () -> (i64) {
block0():
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 0
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
}
"#;

#[test]
fn caller_frame_root_across_call() {
    // The callee returns its own total; the caller adds vroot. Soundness check reads the buffer, so
    // the returned value isn't the total here — read the buffer directly instead.
    let m = parse_module(CALLER_FRAME).expect("parse");
    let init = vec![0u8; 4096];
    let mut f_tw = 1_000_000u64;
    let (_tw_res, tw_snap) =
        run_capture_reserved(&m, 0, &[], &mut f_tw, &init, DEFAULT_RESERVED_LOG2);
    let mut f_bc = 1_000_000u64;
    let (_bc_res, bc_snap) = bytecode::compile_and_run_capture(&m, 0, &[], &mut f_bc, &init)
        .expect("bytecode supports gc.roots");
    // The buffer holds the ascending roots; both must contain the caller's 5000 and the callee's 4096.
    // Read a generous prefix and collect in-range words.
    let collect = |snap: &[u8]| -> BTreeSet<u64> {
        (0..64)
            .map(|i| u64::from_le_bytes(snap[i * 8..i * 8 + 8].try_into().unwrap()))
            .filter(|w| (4096..8192).contains(w))
            .collect()
    };
    let tw = collect(&tw_snap);
    let bc = collect(&bc_snap);
    assert!(
        tw.contains(&5000) && tw.contains(&4096),
        "tw caller-frame scan: {tw:?}"
    );
    assert!(
        tw.is_subset(&bc),
        "bytecode missed a caller-frame root\n  tw={tw:?}\n  bc={bc:?}"
    );
    assert!(
        bc.contains(&5000),
        "bytecode missed the caller's root 5000: {bc:?}"
    );
}

/// A root held in a **parked fiber** must be enumerated — `gc.roots` scans every fiber of the vCPU,
/// not just the calling stack (else a guest GC would free objects a suspended green thread still
/// holds). The fiber computes `5000` (live across its `suspend`), parks, then the root computation
/// runs `gc.roots`; both backends report `{4096 (caller), 5000 (parked fiber)}`.
const PARKED_FIBER: &str = r#"memory 16
func () -> (i64) {
block0():
  vf = ref.func 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  varg = i64.const 0
  vst, vval = cont.resume vk varg
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 0
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  vroot = i64.const 5000
  vy = i64.const 1
  vr = suspend vy
  vsum = i64.add vroot vr
  return vsum
}
"#;

#[test]
fn parked_fiber_root_is_enumerated() {
    check(PARKED_FIBER, 4096, 8192, &[4096, 5000]);
}

/// Security: a **fold-down** mask (clears bits below the top byte) is rejected with `Malformed` on
/// both backends — it could fold a host pointer into the guest window past the range filter (GC.md
/// §3/§6).
const FOLD_DOWN_MASK: &str = r#"memory 16
func () -> (i64) {
block0():
  va = i64.const 5000
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const 72057594037927680
  vbuf = i64.const 0
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
}
"#;

#[test]
fn fold_down_mask_rejected_identically() {
    // vmask = 0x00FF_FFFF_FFFF_FF00 (clears the low byte too) → Malformed on both backends.
    let m = parse_module(FOLD_DOWN_MASK).expect("parse");
    let mut f_tw = 1_000_000u64;
    let tw = svm_interp::run(&m, 0, &[], &mut f_tw);
    let mut f_bc = 1_000_000u64;
    let bc = bytecode::compile_and_run(&m, 0, &[], &mut f_bc).expect("bytecode supports gc.roots");
    assert_eq!(
        tw,
        Err(Trap::Malformed),
        "tree-walker must reject fold-down mask"
    );
    assert_eq!(
        bc,
        Err(Trap::Malformed),
        "bytecode must reject fold-down mask"
    );
}
