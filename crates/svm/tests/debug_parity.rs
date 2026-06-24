//! G1 — cross-engine **source-location parity** (DEBUGGING.md §1b). One `-g` module, three engines,
//! one op→line mapping: the tree-walker (`source_loc` over the executed `IrPc`s), the bytecode engine
//! (`ir_trace` + `source_loc`), and the JIT (`src_ranges`/`symbolize`) must agree. The §6 debug-info
//! "narrow waist" makes them agree *by construction* — each engine threads the same `debug.loc` rows
//! — so this asserts that agreement **directly**. A drift in any one engine's debug-info threading
//! (the interpreter's nearest-preceding lookup, the bytecode `pc→IrPc` map, or the JIT's Cranelift
//! `SourceLoc` baking) fails here instead of silently diverging the debugger view per backend.

use std::collections::BTreeSet;

use svm_interp::{bytecode, source_loc, Inspector, IrPc, Stop, Trap, Value, VarValue};
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

// ---- G2: variable-value parity + the debug/fast delegation boundary -------------------------------

// A window-located source variable `x` at the arg pointer (`v0 + 0`), written twice: 0 -> 11 -> 22.
// Both engines drive the *same* `Mem`, so a window variable has a comparable value at every step. (An
// SSA-located variable is *also* comparable — the bytecode engine gives each value a stable unique
// slot, see `ssa_var_value_parity_per_step` — this fixture just exercises the window path.)
const WINDOW_VAR_DBG: &str = r#"
memory 17
func (i64) -> (i32) {
block0(v0: i64):
  v1 = i32.const 11
  i32.store v0 v1
  v2 = i32.const 22
  i32.store v0 v2
  v3 = i32.const 0
  return v3
}

debug.file 0 "win.c"
debug.fname 0 "w"
debug.type 0 base "int" signed 4
debug.var 0 "x" win 0 "int" 0
debug.loc 0 0 1 0 2 1
debug.loc 0 0 3 0 3 1
"#;

/// The inspection-parity guarantee G2 is about: a *window* source variable holds the **same value at
/// every step** on both engines. The tree-walker reads it through the real debugger APIs
/// (`var_addr` + `read_var`) at each `seek(t)`; the bytecode engine snapshots the same window range at
/// each single-step (`ir_window_trace`). The `(IrPc, bytes)` sequences must be identical — so what the
/// debugger shows for `x` is exactly what the fast engine computes, op for op.
#[test]
fn window_var_value_parity_per_step() {
    let m = parse_module(WINDOW_VAR_DBG).expect("parse");
    let addr: u64 = 1024;
    let width = 4usize;

    // Tree-walker: read `x` via `var_addr`/`read_var` at every executed instruction.
    let mut insp = Inspector::attach(&m, 0, &[Value::I64(addr as i64)], 100_000);
    let mut tw: Vec<(IrPc, Vec<u8>)> = Vec::new();
    let mut t = 0u64;
    let tw_res = loop {
        match insp.seek(t) {
            Stop::Break { pc, .. } => {
                assert_eq!(
                    insp.var_addr(0, "x"),
                    Some(addr),
                    "x resolves to the arg window address"
                );
                let bytes = match insp.read_var(0, "x", width) {
                    Some(VarValue::Bytes(b)) => b,
                    other => panic!("x reads as window bytes, got {other:?}"),
                };
                tw.push((pc, bytes));
                t += 1;
            }
            Stop::Finished(r) => break r,
            Stop::Blocked => panic!("single-threaded fixture must not block"),
        }
    };

    // Bytecode engine: the per-step snapshot of the same window range (declining here would mean a
    // fallback — `expect` pins that the bytecode engine actually ran the module).
    let mut fuel = 100_000u64;
    let (bc, bc_res) =
        bytecode::ir_window_trace(&m, 0, &[Value::I64(addr as i64)], &mut fuel, addr, width)
            .expect("bytecode engine runs this module (not a fallback)");

    assert_eq!(
        bc, tw,
        "window-variable value sequence diverges between the engines"
    );
    assert_eq!(bc_res, tw_res, "result diverges between the engines");

    // Guard against a vacuous pass: `x` must actually change (0 -> 11 -> 22) over the run.
    let distinct: BTreeSet<Vec<u8>> = tw.iter().map(|(_, b)| b.clone()).collect();
    assert!(
        distinct.len() >= 3,
        "fixture must mutate x so the per-step check is meaningful: {distinct:?}"
    );
}

// A thread-spawning module: outside the bytecode engine's *single-vCPU* debug scope. (Production runs
// threads on the bytecode engine — see `bytecode_threads.rs` — but its debug-trace path is single-vCPU
// and declines, delegating to the tree-walker / Milestone-B scheduled Inspector.)
const THREADS_DBG: &str = r#"
func () -> (i64) {
block0():
  vsp = i64.const 0
  va = i64.const 1
  vh = thread.spawn 1 vsp va
  vj = thread.join vh
  vr = i64.const 0
  return vr
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  vz = i64.const 0
  return vz
}
"#;

/// The delegation boundary G2 names, pinned: the bytecode engine's single-vCPU debug-trace path
/// **declines** (`None`) a module outside its seam-free scope, so a debugger must fall back to the
/// tree-walker. If a later slice extended the bytecode debug scope to threads, this `None` would flip
/// — and that is exactly the boundary change worth noticing here.
#[test]
fn bytecode_debug_trace_declines_outside_single_vcpu_scope() {
    let m = parse_module(THREADS_DBG).expect("parse");
    let mut fuel = 100_000u64;
    assert!(
        bytecode::ir_trace(&m, 0, &[], &mut fuel).is_none(),
        "the bytecode debug-trace declines a multi-vCPU module (delegates to the tree-walker)"
    );
    assert!(
        bytecode::ir_window_trace(&m, 0, &[], &mut fuel, 0, 4).is_none(),
        "the window-variable trace declines it too"
    );
}

// A single-block function with two SSA-located source variables: `a = v2` (x+1) and `b = v3` (a*a).
// The bytecode engine gives each value a stable unique slot (no register reuse), so `v2`/`v3` are
// directly inspectable there — `regs[base + i]` — exactly the storage the tree-walker indexes.
const SSA_VAR_DBG: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.add v0 v1
  v3 = i32.mul v2 v2
  v4 = i32.sub v3 v1
  return v4
}

debug.file 0 "ssa.c"
debug.fname 0 "f"
debug.type 0 base "int" signed 4
debug.var 0 "a" ssa 2 "int"
debug.var 0 "b" ssa 3 "int"
"#;

/// SSA-value inspection parity (the correction to the earlier "precluded by design" claim): an
/// SSA-located variable holds the **same typed value on both engines at every step**. The bytecode
/// engine assigns a stable unique slot per value (no register reuse), so it is just as inspectable as
/// the tree-walker here. Compares the per-step block-local value vectors op-for-op, then reads the
/// named variables `a`/`b` through the real debugger API (`read_var`) and checks them against the
/// bytecode slot values.
#[test]
fn ssa_var_value_parity_per_step() {
    let m = parse_module(SSA_VAR_DBG).expect("parse");
    let args = [Value::I32(5)];

    // Bytecode engine: per-step typed SSA values (declining would be a fallback — `expect` pins it ran).
    let mut fuel = 100_000u64;
    let (bc, bc_res) = bytecode::ir_value_trace(&m, 0, &args, &mut fuel)
        .expect("bytecode engine runs this single-block module");

    // Tree-walker: per-step *defined* block-local values via `read_ir_value` (the debugger API).
    let mut insp = Inspector::attach(&m, 0, &args, 100_000);
    let mut tw: Vec<(IrPc, Vec<Value>)> = Vec::new();
    let mut t = 0u64;
    let tw_res = loop {
        match insp.seek(t) {
            Stop::Break { pc, .. } => {
                let mut vals = Vec::new();
                let mut i = 0usize;
                while let Some(v) = insp.read_ir_value(0, i) {
                    vals.push(v);
                    i += 1;
                }
                tw.push((pc, vals));
                t += 1;
            }
            Stop::Finished(r) => break r,
            Stop::Blocked => panic!("single-threaded fixture must not block"),
        }
    };

    assert_eq!(bc.len(), tw.len(), "step count diverges");
    assert_eq!(bc_res, tw_res, "result diverges");
    for (k, ((bpc, bvals), (tpc, tvals))) in bc.iter().zip(&tw).enumerate() {
        assert_eq!(bpc, tpc, "step {k}: IrPc diverges");
        // The tree-walker exposes only the values defined so far; they must equal the bytecode engine's
        // slots over that prefix (later slots hold not-yet-computed defaults the debugger doesn't show).
        assert_eq!(
            &bvals[..tvals.len()],
            &tvals[..],
            "step {k}: SSA values diverge between engines"
        );
    }

    // The debugger's variable API over the SSA locations: `a = v2 = 6`, `b = v3 = 36`, both defined at
    // the last recorded step (before `v4`). They must match the bytecode slots 2 and 3.
    let last = tw.len() - 1;
    assert_eq!(tw[last].1.len(), 4, "v0..v3 defined at the last step");
    let mut insp2 = Inspector::attach(&m, 0, &args, 100_000);
    insp2.seek(last as u64);
    assert_eq!(
        insp2.read_var(0, "a", 4),
        Some(VarValue::Value(bc[last].1[2])),
        "read_var(a) matches the bytecode slot for v2"
    );
    assert_eq!(
        insp2.read_var(0, "b", 4),
        Some(VarValue::Value(bc[last].1[3])),
        "read_var(b) matches the bytecode slot for v3"
    );
    // Concretely: x=5 -> a=6, b=36.
    assert_eq!(bc[last].1[2], Value::I32(6));
    assert_eq!(bc[last].1[3], Value::I32(36));
}

// ---- G3 foundation: runtime breakpoint parity on a *second* engine -------------------------------

// A counting loop with two SSA loop variables visible in the body block: `i` = block1's value 0 (the
// down-counter), `acc` = block1's value 1 (the accumulator). Stopping at the body each iteration lets
// us compare the inspected (i, acc) across engines hit-for-hit.
const LOOP_VAR_DBG: &str = r#"
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

debug.file 0 "loopvar.c"
debug.fname 0 "loopsum"
debug.type 0 base "int" signed 4
debug.var 0 "i" ssa 0 "int"
debug.var 0 "acc" ssa 1 "int"
"#;

/// G3 foundation — **runtime** debug parity on a second engine, the piece G1/G2's one-shot traces
/// don't exercise: the tree-walker `Inspector` and the resumable `bytecode::DebugRun` are driven
/// through the *same loop-body breakpoint* and must report identical stop locations and inspected
/// `(i, acc)` at **every hit** (resume included), plus the same result. This is the load-bearing
/// behavior a DAP-over-bytecode backend would wire into.
#[test]
fn breakpoint_runtime_parity_across_loop_iterations() {
    let m = parse_module(LOOP_VAR_DBG).expect("parse");
    let args = [Value::I32(3)];
    // The loop-body breakpoint: block 1, inst 0 (`v4 = add`); block-local 0/1 are `i`/`acc`.
    let bp = IrPc {
        module: 0,
        func: 0,
        block: 1,
        inst: 0,
    };

    // Tree-walker: stop at the breakpoint each iteration; read i/acc via the inspection API.
    let mut insp = Inspector::attach(&m, 0, &args, 100_000);
    insp.set_breakpoint(bp);
    let mut tw_hits: Vec<(Value, Value)> = Vec::new();
    let tw_res = loop {
        match insp.run_until_stop() {
            Stop::Break { pc, .. } => {
                assert_eq!(pc, bp, "tree-walker stopped at the loop-body breakpoint");
                tw_hits.push((
                    insp.read_ir_value(0, 0).unwrap(),
                    insp.read_ir_value(0, 1).unwrap(),
                ));
            }
            Stop::Finished(r) => break r,
            Stop::Blocked => panic!("single-threaded fixture must not block"),
        }
    };

    // Bytecode engine: the resumable debug session, same breakpoint, same inspection.
    let mut dbg = bytecode::DebugRun::new(&m, 0, &args).expect("bytecode debug session");
    let mut fuel = 100_000u64;
    let mut bc_hits: Vec<(Value, Value)> = Vec::new();
    while let Some(pc) = dbg.run_to(&[bp], &mut fuel) {
        assert_eq!(
            pc, bp,
            "bytecode engine stopped at the loop-body breakpoint"
        );
        bc_hits.push((dbg.value(0).unwrap(), dbg.value(1).unwrap()));
    }
    let bc_res = dbg.result().cloned().unwrap();

    assert_eq!(
        bc_hits, tw_hits,
        "per-iteration (i, acc) at the breakpoint diverge between engines"
    );
    assert_eq!(bc_res, tw_res, "result diverges between engines");
    // Concretely: counter 3 -> 0 yields hits (3,0),(2,3),(1,5) and result 6.
    assert_eq!(
        tw_hits,
        vec![
            (Value::I32(3), Value::I32(0)),
            (Value::I32(2), Value::I32(3)),
            (Value::I32(1), Value::I32(5)),
        ]
    );
    assert_eq!(tw_res, Ok(vec![Value::I32(6)]));
}
