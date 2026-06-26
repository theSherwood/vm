//! §5 W3 — the **interpreter trap-time backtrace** (`run_traced` / `run_with_host_traced` +
//! `source_loc`), the cross-platform counterpart to the JIT's `last_trap_backtrace`. The interpreter
//! reifies its call stack as `Vec<Frame>` and does *not* unwind it on a trap, so the executor
//! snapshots the live frames into the run outcome; this exposes them. Useful for kill diagnostics and
//! for the interp↔JIT differential fuzzer to report *where* a trap/divergence occurred.

use svm_interp::{func_name, run_traced, source_loc, Trap, Value};
use svm_text::parse_module;

/// A run's result plus its trap-time backtrace as `(name, file, line)` per frame (innermost first),
/// where `name` is the `-g` function name (or `fn{N}` when unnamed).
type TracedRun = (Result<Vec<Value>, Trap>, Vec<(String, String, u32)>);

/// `(name, file, line)` for each frame of the trap-time backtrace, innermost first — `IrPc`s resolved
/// to source + function name via the module's `-g` debug info.
fn traced(src: &str, arg: Value) -> TracedRun {
    let m = parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut fuel = u64::MAX;
    let (res, bt, _) = run_traced(&m, 0, &[arg], &mut fuel);
    let frames = bt
        .iter()
        .map(|&pc| {
            let s = source_loc(&m, pc).expect("each guest frame resolves to source under -g");
            let name = func_name(&m, pc.func)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("fn{}", pc.func));
            (name, s.file, s.line)
        })
        .collect();
    (res, frames)
}

/// An out-of-bounds store traps `MemoryFault`; the backtrace names the storing function + line.
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

/// func0 calls func1, which divides by zero — a two-frame explicit-check trap. The call's result
/// feeds a later add, so func0's frame is live (not a tail call). call @ line 40; div @ line 30.
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
fn interp_trap_backtrace_names_the_faulting_store() {
    let (res, bt) = traced(STORE_OOB, Value::I32(0));
    assert!(matches!(res, Err(Trap::MemoryFault)), "got {res:?}");
    assert_eq!(
        bt,
        vec![("store_oob".into(), "fault.c".into(), 10)],
        "one frame, named, at the store line"
    );
}

#[test]
fn interp_trap_backtrace_walks_the_caller_chain() {
    let (res, bt) = traced(CALL_THEN_DIV, Value::I32(7));
    assert!(matches!(res, Err(Trap::DivByZero)), "got {res:?}");
    assert_eq!(
        bt,
        vec![
            ("divz".into(), "div.c".into(), 30),
            ("outer".into(), "div.c".into(), 40)
        ],
        "innermost callee (divz@div) then caller (outer@call), by name"
    );
}

#[test]
fn a_clean_run_has_no_trap_backtrace() {
    let src = "\
func (i32) -> (i32) {
block0(v0: i32):
  return v0
}
";
    let (res, bt) = traced(src, Value::I32(42));
    assert_eq!(res, Ok(vec![Value::I32(42)]));
    assert!(
        bt.is_empty(),
        "a non-trapping run leaves an empty backtrace"
    );
}
