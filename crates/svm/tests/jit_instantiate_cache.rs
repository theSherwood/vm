//! PROCESS.md S1 — the JIT **per-carve compile cache**. The JIT re-compiles a §14 child as a
//! top-level guest over its own window; `compile_child` bakes only the *size* mask and the window
//! **base is a runtime arg** to `run_guarded`, so one compiled child runs at any carve offset. This
//! test pins that repeat spawns of the same `(module, entry, size)` — even at *different* offsets —
//! compile the child **once** (`svm_jit::child_compiles()` advances by 1, not 2), while each spawn
//! still runs correctly confined to its own carve. This attacks the F6 / gap-4 cost the design
//! flagged: without the cache, a shell spawning the same applet N times would run Cranelift N times.
//!
//! Sole test in its own binary so the process-wide compile counter is stable (cargo runs each test
//! binary in its own process; a lone test rules out interference from siblings in the same binary).

use svm_interp::Host;
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

/// Parent (func 0) instantiates the child (func 1) in a 4 KiB carve at `off_a`, joins it, then does
/// the **same** at `off_b`, and returns the sum of the two results. The child stores `42` at its own
/// offset 0 (→ the carve base in the parent window) and returns `7`, so a correct confined run leaves
/// `42` at each carve base and the parent returns `14`.
const PARENT: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  v1 = i64.const 1\n\
  v2 = i64.const 65536\n\
  v3 = i64.const 12\n\
  v4 = i64.const 0\n\
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
  v7 = i64.const 69632\n\
  v8 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v7, v3, v4)\n\
  v9 = cap.call 6 1 (i32) -> (i64) v0 (v8)\n\
  v10 = i64.add v6 v9\n\
  return v10\n\
  }\n\
}\n\
func (i64) -> (i64) {\n\
block 0 (v0: i64) {\n\
  v1 = i64.const 0\n\
  v2 = i32.const 42\n\
  i32.store8 v1 v2\n\
  v3 = i64.const 7\n\
  return v3\n\
  }\n\
}\n";

#[test]
fn same_child_at_two_offsets_compiles_once() {
    let m = parse_module(PARENT).expect("parse");
    verify_module(&m).expect("verify");
    let win = 1u64 << 17;
    let init = vec![0u8; win as usize];

    let mut host = Host::new();
    let h = host.grant_instantiator(0, win);

    let before = svm_jit::child_compiles();
    let (jo, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[h as i64],
        &init,
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit run");
    let compiled = svm_jit::child_compiles() - before;

    // Both children ran and returned 7 → sum 14.
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[14]),
        "expected both children to run (sum 14), got {jo:?}"
    );
    // Each ran confined to its own carve: the `42` marker landed at each carve base.
    assert_eq!(jmem[65536], 42, "child A store missing at carve A base");
    assert_eq!(jmem[69632], 42, "child B store missing at carve B base");
    // Confinement: nothing outside the two carves' first byte was touched (both carves start zeroed;
    // the child writes only offset 0). Spot-check the byte just past carve A's marker and a byte in
    // between stays zero.
    assert_eq!(jmem[65537], 0, "child A escaped past its marker");
    assert_eq!(jmem[0], 0, "a child escaped into the parent's low window");

    // The load-bearing assertion: the child compiled **once**, then the second spawn (a different
    // offset) reused the cached code — the base is a runtime arg, so position-independent reuse.
    assert_eq!(
        compiled, 1,
        "same (module, entry, size) spawned twice must JIT-compile once, saw {compiled}"
    );
}
