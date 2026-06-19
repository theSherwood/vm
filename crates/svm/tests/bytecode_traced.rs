//! Parity harness for the bytecode engine's **trap-time backtrace** (the mirror of the tree-walker's
//! `run_with_host_traced` / `run_traced`): on a trap the engine reifies the same call stack — as
//! [`IrPc`]s, innermost frame first — that the tree-walker snapshots from its `Vec<Frame>`.
//!
//! The tree-walker is the reference oracle. For every trapping program in the engine's single-vCPU
//! subset this asserts the two backends agree on **(a)** the result, **(b)** the raw `IrPc` backtrace,
//! and **(c)** that backtrace resolved to source `(name, file, line)` via `-g` debug info — and that
//! the bytecode engine actually *drove* the run (did not silently fall back). That equality is what
//! lets a kill diagnostic name the same program point regardless of which backend executed the guest.

use svm_interp::{bytecode, func_name, run_with_host_traced, source_loc, Host, IrPc, Value};
use svm_ir::Module;
use svm_text::parse_module;

/// Resolve a raw backtrace to `(name, file, line)` per frame (innermost first) — the rendered form a
/// kill diagnostic shows; `name` is the `-g` function name, or `fn{N}` when unnamed.
fn render(m: &Module, bt: &[IrPc]) -> Vec<(String, String, u32)> {
    bt.iter()
        .map(|&pc| {
            let s = source_loc(m, pc).expect("each guest frame resolves to source under -g");
            let name = func_name(m, pc.func)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("fn{}", pc.func));
            (name, s.file, s.line)
        })
        .collect()
}

/// Assert the bytecode engine's traced run is parity-equal to the tree-walker's, at every level:
/// result, raw `IrPc` backtrace, and resolved source. Also asserts the bytecode engine drove the run
/// (the traced entry returned `Some`) — otherwise the "parity" would be vacuous (a silent fallback to
/// the very oracle we compare against).
fn check(src: &str, arg: Value) {
    check_fuel(src, arg, u64::MAX);
}

/// [`check`], but with an explicit `fuel` budget — so a small budget exercises the [`Trap::OutOfFuel`]
/// path (the one trap the tree-walker raises *before* advancing `inst`, so the innermost frame is the
/// faulting op itself, not one past it).
fn check_fuel(src: &str, arg: Value, fuel: u64) {
    let m = parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");

    let mut tw_fuel = fuel;
    let (tw_res, tw_bt) = run_with_host_traced(&m, 0, &[arg], &mut tw_fuel, &mut Host::new());

    let mut bc_fuel = fuel;
    let (bc_res, bc_bt) =
        bytecode::compile_and_run_with_host_traced(&m, 0, &[arg], &mut bc_fuel, &mut Host::new())
            .expect("bytecode engine must drive this single-vCPU module");

    assert_eq!(tw_res, bc_res, "result: tree-walker != bytecode\n{src}");
    assert_eq!(
        tw_bt, bc_bt,
        "raw IrPc backtrace: tree-walker != bytecode\n{src}"
    );
    assert_eq!(
        render(&m, &tw_bt),
        render(&m, &bc_bt),
        "resolved backtrace: tree-walker != bytecode\n{src}"
    );
}

/// An out-of-bounds store traps `MemoryFault`; one frame, named, at the store line.
const STORE_OOB: &str = "\
memory 16
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 65532
  v2 = i64.const 0
  i64.store v1 v2
  v3 = i32.const 0
  return v3
}
debug.file 0 \"fault.c\"
debug.fname 0 \"store_oob\"
debug.loc 0 0 2 0 10 3
";

#[test]
fn single_frame_memory_fault() {
    check(STORE_OOB, Value::I32(0));
}

/// func0 calls func1, which divides by zero — a two-frame, non-tail trap (the call result feeds a
/// later add, so func0's window stays live). The backtrace must walk callee then caller.
const CALL_THEN_DIV: &str = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = call 1 (v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = i32.div_s v0 v1
  return v2
}
debug.file 0 \"div.c\"
debug.fname 0 \"outer\"
debug.fname 1 \"divz\"
debug.loc 0 0 0 0 40 5
debug.loc 1 0 1 0 30 3
";

#[test]
fn two_frame_caller_chain() {
    check(CALL_THEN_DIV, Value::I32(7));
}

/// Three frames deep: func0 → func1 → func2, the innermost dividing by zero. Pins that every caller
/// window on the reified `stack` (not just the top one) lands at its own call site.
const DEEP_CHAIN: &str = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = call 1 (v0)
  return v1
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 5
  v2 = i32.add v0 v1
  v3 = call 2 (v2)
  return v3
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = i32.rem_s v0 v1
  return v2
}
debug.file 0 \"chain.c\"
debug.fname 0 \"top\"
debug.fname 1 \"mid\"
debug.fname 2 \"bot\"
debug.loc 0 0 0 0 10 3
debug.loc 1 0 2 0 20 3
debug.loc 2 0 1 0 30 3
";

#[test]
fn three_frame_caller_chain() {
    check(DEEP_CHAIN, Value::I32(9));
}

/// The trapping call sits mid-block with live instructions after it, so the caller's recorded location
/// is the call site, not the post-call op — a `inst + 1` boundary check against the tree-walker.
const CALL_MIDBLOCK: &str = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 100
  v2 = call 1 (v0)
  v3 = i32.add v2 v1
  return v3
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 0
  v2 = i32.div_s v0 v1
  return v2
}
debug.file 0 \"mid.c\"
debug.fname 0 \"caller\"
debug.fname 1 \"callee\"
debug.loc 0 0 1 0 11 3
debug.loc 1 0 1 0 22 3
";

#[test]
fn caller_location_is_the_call_site() {
    check(CALL_MIDBLOCK, Value::I32(3));
}

/// A trap *at a terminator* (`unreachable`) — not an instruction. The faulting op has no `(block,
/// inst)` of its own (terminators are non-steppable), so the engine reports the block's instruction
/// count as the `inst`, exactly the position the tree-walker's frame carries. The block here has live
/// instructions before the terminator, pinning that count.
const UNREACHABLE: &str = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 7
  v2 = i32.add v0 v1
  unreachable
}
debug.file 0 \"halt.c\"
debug.fname 0 \"halt\"
debug.loc 0 0 1 0 9 3
";

#[test]
fn terminator_trap_unreachable() {
    check(UNREACHABLE, Value::I32(1));
}

/// `unreachable` as the *only* op of a block reached by a branch — a zero-instruction block, so the
/// reported `inst` is 0. Guards the terminator-location path against the empty-block corner a
/// scan-back heuristic would get wrong.
const UNREACHABLE_EMPTY_BLOCK: &str = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.eqz v0
  br_if v1 block1(v0) block2(v0)
block1(v2: i32):
  unreachable
block2(v3: i32):
  return v3
}
debug.file 0 \"br.c\"
debug.fname 0 \"maybe_halt\"
debug.loc 0 0 0 0 3 3
debug.loc 0 1 0 0 5 3
";

#[test]
fn terminator_trap_in_empty_block() {
    check(UNREACHABLE_EMPTY_BLOCK, Value::I32(0)); // takes block1 → unreachable
}

/// A clean run leaves an empty backtrace on both backends (and the same result).
const CLEAN: &str = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
}
debug.file 0 \"ok.c\"
debug.fname 0 \"ok\"
debug.loc 0 0 0 0 1 3
";

/// Straight-line arithmetic lowered one op per IR instruction, so per-op fuel charging matches across
/// engines: a small budget runs out mid-block and both backends raise `OutOfFuel` at the same op —
/// the case that pins the `bump = 0` (no innermost advance) branch of the backtrace.
const FUEL_BURN: &str = "\
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 1
  v2 = i32.add v0 v1
  v3 = i32.add v2 v1
  v4 = i32.add v3 v1
  v5 = i32.add v4 v1
  return v5
}
debug.file 0 \"burn.c\"
debug.fname 0 \"burn\"
debug.loc 0 0 0 0 5 3
debug.loc 0 0 2 0 7 3
";

#[test]
fn out_of_fuel_backtrace_matches() {
    // Enough fuel for a few ops but not the whole block — traps `OutOfFuel` partway through.
    for fuel in 1..=4 {
        check_fuel(FUEL_BURN, Value::I32(0), fuel);
    }
}

#[test]
fn clean_run_has_empty_backtrace() {
    check(CLEAN, Value::I32(41));
    // belt-and-braces: the rendered backtrace really is empty (not just equal-and-nonempty)
    let m = parse_module(CLEAN).expect("parse");
    let mut fuel = u64::MAX;
    let (res, bt) =
        bytecode::compile_and_run_with_host_traced(&m, 0, &[Value::I32(41)], &mut fuel, &mut Host::new())
            .expect("drove");
    assert_eq!(res, Ok(vec![Value::I32(42)]));
    assert!(bt.is_empty(), "clean run ⇒ empty backtrace");
}
