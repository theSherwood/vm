//! G1 — cross-engine **source-location parity** (DEBUGGING.md §1b). One `-g` module, three engines,
//! one op→line mapping: the tree-walker (`source_loc` over the executed `IrPc`s), the bytecode engine
//! (`ir_trace` + `source_loc`), and the JIT (`src_ranges`/`symbolize`) must agree. The §6 debug-info
//! "narrow waist" makes them agree *by construction* — each engine threads the same `debug.loc` rows
//! — so this asserts that agreement **directly**. A drift in any one engine's debug-info threading
//! (the interpreter's nearest-preceding lookup, the bytecode `pc→IrPc` map, or the JIT's Cranelift
//! `SourceLoc` baking) fails here instead of silently diverging the debugger view per backend.

use std::collections::BTreeSet;

use svm_interp::{bytecode, source_loc, Inspector, IrPc, Stop, Trap, Value};
use svm_ir::{Module, DEFAULT_RESERVED_LOG2};
use svm_jit::{CompiledModule, Quota, INERT_CAP_THUNK};
use svm_text::parse_module;

/// The tree-walker's executed-instruction `IrPc` sequence, via logical-time `seek` (terminators don't
/// tick, so each `t` is one executed instruction), plus the run result — the reference the other two
/// engines are checked against.
fn tw_trace(m: &Module, args: &[Value], fuel: u64) -> (Vec<IrPc>, Result<Vec<Value>, Trap>) {
    let mut insp = Inspector::attach(m, 0, args, fuel);
    let mut pcs = Vec::new();
    let mut t = 0u64;
    loop {
        match insp.seek(t) {
            Stop::Break { pc, .. } => {
                pcs.push(pc);
                t += 1;
            }
            Stop::Finished(r) => return (pcs, r),
            Stop::Blocked => panic!("single-threaded parity fixture must not block"),
        }
    }
}

/// Map an executed `IrPc` sequence to its source lines, dropping ops with no `debug.loc`. A `const`
/// before the first loc has none — the interpreter's nearest-preceding lookup returns `None` and the
/// JIT filters its default `SourceLoc`, so both sides agree to exclude it.
fn lines_of(m: &Module, pcs: &[IrPc]) -> Vec<u32> {
    pcs.iter()
        .filter_map(|pc| source_loc(m, *pc).map(|s| s.line))
        .collect()
}

fn compile_jit(m: &Module) -> CompiledModule {
    svm_verify::verify_module(m).expect("verify");
    CompiledModule::compile(
        m,
        0,
        INERT_CAP_THUNK,
        core::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("jit compiles")
}

/// The set of source lines the JIT's finalized machine-address map attributes to function 0.
fn jit_lineset(cm: &CompiledModule) -> BTreeSet<u32> {
    cm.src_ranges()
        .iter()
        .filter(|r| r.func == 0)
        .map(|r| r.line)
        .collect()
}

/// Assert the three engines agree on the op→line mapping for `func 0` run on `args`.
///
/// `exact_jit`: when every located op executes (straight-line / full loop coverage) the JIT's line
/// set equals the executed set; for a branch fixture only a subset runs, but the not-taken arm's
/// lines still live in the JIT map — so the executed lines must be a **subset** of the JIT's.
fn check(name: &str, src: &str, args: &[Value], exact_jit: bool) {
    let m = parse_module(src).expect("parse");

    // Tree-walker (reference) vs bytecode engine: identical executed-op sequence and result.
    let (tw_pcs, tw_res) = tw_trace(&m, args, 100_000);
    let mut fuel = 100_000u64;
    let (bc_pcs, bc_res) = bytecode::ir_trace(&m, 0, args, &mut fuel)
        .unwrap_or_else(|| panic!("{name}: bytecode engine declined the run (hit a seam?)"));
    assert_eq!(
        bc_pcs, tw_pcs,
        "{name}: bytecode IrPc sequence != tree-walker"
    );
    assert_eq!(bc_res, tw_res, "{name}: bytecode result != tree-walker");

    // The executed source-line sequence both interpreters report (op-for-op, not just the set).
    let tw_lines = lines_of(&m, &tw_pcs);
    assert_eq!(
        lines_of(&m, &bc_pcs),
        tw_lines,
        "{name}: bytecode source-line sequence != tree-walker"
    );
    let executed: BTreeSet<u32> = tw_lines.iter().copied().collect();
    assert!(
        !executed.is_empty(),
        "{name}: fixture must execute located ops"
    );

    // JIT: its source map (built from the same §6 locs, threaded through Cranelift) covers exactly —
    // or, for a branch, a superset of — the lines the interpreters step through.
    let cm = compile_jit(&m);
    let jit = jit_lineset(&cm);
    if exact_jit {
        assert_eq!(
            jit, executed,
            "{name}: JIT line set != interpreter-executed lines"
        );
    } else {
        assert!(
            executed.is_subset(&jit),
            "{name}: executed lines {executed:?} not subset of JIT {jit:?}"
        );
    }

    // Each JIT range round-trips through `symbolize` (machine-pc → source) to its own line — the
    // resolution the trap symbolizer and DWARF emitter rely on.
    for r in cm.src_ranges().iter().filter(|r| r.func == 0) {
        let loc = cm
            .symbolize(r.lo as usize)
            .unwrap_or_else(|| panic!("{name}: symbolize {:#x}", r.lo));
        assert_eq!(loc.line, r.line, "{name}: symbolize line != range line");
    }
}

// Straight-line: three computing ops on lines 2/3/4 (the leading consts inherit nearest-preceding /
// carry no loc). Every op runs, so all three engines map to exactly {2,3,4}.
const COMPUTE_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.add v0 v1
  v3 = i32.const 3
  v4 = i32.mul v2 v3
  v5 = i32.const 2
  v6 = i32.sub v4 v5
  return v6
}

debug.file 0 "compute.c"
debug.fname 0 "compute"
debug.loc 0 0 1 0 2 7
debug.loc 0 0 3 0 3 7
debug.loc 0 0 5 0 4 3
"#;

// A counting loop whose two located ops are the body's `add`s (lines 2/3) — *computing* ops that each
// emit machine code, so the JIT maps both. The lines repeat in the executed sequence, exercising
// op-for-op (not just set) parity between the two interpreters. (Locating on the `add`s, not the
// folded `const`, keeps the JIT's set exactly equal — see `jit_elides_const_only_source_line`.)
const LOOP_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  br block1(v0, v1)
block1(v2: i32, v3: i32):
  v4 = i32.add v3 v2
  v5 = i32.const -1
  v6 = i32.add v2 v5
  br_if v6 block1(v6, v4) block2(v4)
block2(v7: i32):
  return v7
}

debug.file 0 "loop.c"
debug.fname 0 "loop_sum"
debug.loc 0 1 0 0 2 1
debug.loc 0 1 2 0 3 1
"#;

// A two-armed branch: the taken arm's line is stepped, but the JIT's map carries both arms (lines 3
// and 5). So the executed line set is a strict subset of the JIT's — the `exact_jit = false` case.
const BRANCH_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  br_if v0 block1() block2()
block1():
  v1 = i32.const 10
  return v1
block2():
  v2 = i32.const 20
  return v2
}

debug.file 0 "branch.c"
debug.fname 0 "branch"
debug.loc 0 1 0 0 3 1
debug.loc 0 2 0 0 5 1
"#;

#[test]
fn parity_straight_line() {
    check("compute", COMPUTE_DBG, &[Value::I32(5)], true);
}

#[test]
fn parity_loop_repeated_lines() {
    check("loop", LOOP_DBG, &[Value::I32(4)], true);
}

#[test]
fn parity_branch_taken_arm_is_subset_of_jit_map() {
    // Either arm: the executed line is one of the JIT's two, never outside it.
    check("branch_then", BRANCH_DBG, &[Value::I32(1)], false);
    check("branch_else", BRANCH_DBG, &[Value::I32(0)], false);
}

// Line 2 sits *only* on a `const` op (`v1`), line 3 on the `add` that uses it. The interpreters
// execute the const as a real op and attribute line 2 to it; the JIT folds the single-use const into
// the add's immediate, emitting no machine instruction for it — so line 2 has no source range.
const CONST_ELIDE_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
}

debug.file 0 "ce.c"
debug.loc 0 0 0 0 2 1
debug.loc 0 0 1 0 3 1
"#;

/// The one **legitimate** cross-engine divergence, pinned so it is not mistaken for a debug-info
/// regression: a source line that belongs only to a folded `const` is stepped by both interpreters
/// but carries no JIT machine-code range (compiled code has no instruction for a materialized
/// immediate). Invariant that must still hold: every line the JIT *does* map is one the interpreters
/// step — the JIT never invents a line.
#[test]
fn jit_elides_const_only_source_line() {
    let m = parse_module(CONST_ELIDE_DBG).expect("parse");

    let (tw_pcs, _) = tw_trace(&m, &[Value::I32(5)], 100_000);
    let executed: BTreeSet<u32> = lines_of(&m, &tw_pcs).iter().copied().collect();
    assert!(
        executed.contains(&2) && executed.contains(&3),
        "interpreters step both the const line (2) and the add line (3): {executed:?}"
    );

    let cm = compile_jit(&m);
    let jit = jit_lineset(&cm);
    assert!(
        !jit.contains(&2),
        "the folded const's line (2) has no JIT machine range: {jit:?}"
    );
    assert!(jit.contains(&3), "the add's line (3) is mapped: {jit:?}");
    assert!(
        jit.is_subset(&executed),
        "the JIT never maps a line the interpreters don't step: jit={jit:?} executed={executed:?}"
    );
}
