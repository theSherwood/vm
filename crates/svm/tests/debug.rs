//! Interpreter-rooted debugger (DEBUGGING.md W2/W8, Milestone A slice 1): breakpoints, single
//! stepping, IR-value / window inspection, and a backtrace across a call — driven through the
//! host-side `Inspector`. These pin the S4 per-op seam + S5 driver model end-to-end.

use svm_interp::{
    Host, Inspector, IrPc, SourceLoc, Stop, StopReason, StreamRole, Value, VarValue, WatchKind,
};
use svm_text::{parse_module, print_module};

// sum = 1 + 2 + ... + N via a back-edge loop with block parameters (same shape as pipeline.rs).
const LOOP_SUM: &str = r#"
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
"#;

// A callee + caller, to exercise a multi-frame backtrace.
const CALLER: &str = r#"
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 7
  v2 = call 1 (v0)
  return v2
}
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i32.const 10
  v2 = i32.add v0 v1
  return v2
}
"#;

fn finished_ok(stop: Stop) -> Vec<Value> {
    match stop {
        Stop::Finished(Ok(vals)) => vals,
        other => panic!("expected Finished(Ok), got {other:?}"),
    }
}

#[test]
fn runs_to_completion_with_no_breakpoints() {
    // An attached-but-unconstrained run matches a plain `run` (S7: debug off-path is transparent).
    let m = parse_module(LOOP_SUM).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(5)], 1_000_000);
    let out = finished_ok(insp.run_until_stop());
    assert_eq!(out, vec![Value::I32(15)]); // 1+2+3+4+5
    assert_eq!(insp.result(), Some(&Ok(vec![Value::I32(15)])));
    assert!(insp.clock() > 0, "executed ops should advance logical time");
}

#[test]
fn breakpoint_stops_at_the_loop_body_each_iteration() {
    let m = parse_module(LOOP_SUM).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(3)], 1_000_000);
    // Break at the accumulate op (block1, inst 0: `v4 = i32.add v3 v2`).
    let bp = IrPc {
        module: 0,
        func: 0,
        block: 1,
        inst: 0,
    };
    insp.set_breakpoint(bp);

    // N = 3 ⇒ the loop body runs three times, so we hit the breakpoint three times.
    let mut accs = Vec::new();
    for _ in 0..3 {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                pc,
            } => {
                assert_eq!(pc, bp);
                // v2 (counter) and v3 (running accumulator) are this block's params: vals[0], vals[1].
                let v2 = insp.read_ir_value(0, 0).expect("v2");
                let v3 = insp.read_ir_value(0, 1).expect("v3");
                accs.push((v2, v3));
            }
            other => panic!("expected breakpoint, got {other:?}"),
        }
    }
    // Counter walks 3,2,1; accumulator walks 0,3,5.
    assert_eq!(
        accs,
        vec![
            (Value::I32(3), Value::I32(0)),
            (Value::I32(2), Value::I32(3)),
            (Value::I32(1), Value::I32(5)),
        ]
    );
    // After the third hit, removing the breakpoint lets it finish: 1+2+3 = 6.
    assert!(insp.clear_breakpoint(bp));
    let out = finished_ok(insp.run_until_stop());
    assert_eq!(out, vec![Value::I32(6)]);
}

#[test]
fn single_step_advances_exactly_one_op_and_ticks_the_clock() {
    let m = parse_module(LOOP_SUM).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(2)], 1_000_000);

    // block0 has a single instruction (`v1 = i32.const 0`); the `br` is the block *terminator*,
    // not an `insts` op, so it isn't a hookable step point. Stepping one op therefore runs the
    // const and the branch, landing before block1's first op. The clock counts non-terminator ops.
    let before = insp.clock();
    match insp.step() {
        Stop::Break {
            reason: StopReason::Step,
            pc,
        } => {
            assert_eq!(
                pc,
                IrPc {
                    module: 0,
                    func: 0,
                    block: 1,
                    inst: 0
                }
            );
        }
        other => panic!("expected step stop, got {other:?}"),
    }
    assert_eq!(
        insp.clock(),
        before + 1,
        "exactly one (non-terminator) op executed"
    );
    // In block1 the frame's values are its params v2 (counter) and v3 (accumulator) = N, 0.
    assert_eq!(insp.read_ir_value(0, 1), Some(Value::I32(0)));

    // A handful more steps stay single-op; the clock advances monotonically.
    let mut last = insp.clock();
    for _ in 0..4 {
        if let Stop::Break { .. } = insp.step() {
            assert_eq!(insp.clock(), last + 1);
            last = insp.clock();
        } else {
            break;
        }
    }
}

#[test]
fn backtrace_shows_both_frames_inside_the_callee() {
    let m = parse_module(CALLER).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(5)], 1_000_000);
    // Break on the callee's add (func 1, block0, inst1: `v2 = i32.add v0 v1`).
    let bp = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 1,
    };
    insp.set_breakpoint(bp);
    match insp.run_until_stop() {
        Stop::Break { pc, .. } => assert_eq!(pc, bp),
        other => panic!("expected callee breakpoint, got {other:?}"),
    }
    let bt = insp.backtrace();
    assert_eq!(bt.len(), 2, "callee + caller frames");
    // Innermost frame is the callee (func 1); the caller (func 0) is paused at its `call` op.
    assert_eq!(bt[0].pc.func, 1);
    assert_eq!(bt[1].pc.func, 0);
    // The callee's arg arrived as v0 = 5; v1 = 10 was just produced.
    assert_eq!(insp.read_ir_value(0, 0), Some(Value::I32(5)));
    assert_eq!(insp.read_ir_value(0, 1), Some(Value::I32(10)));

    // Let it finish: callee returns 5 + 10 = 15.
    assert!(insp.clear_breakpoint(bp));
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I32(15)]);
}

#[test]
fn reads_window_memory_a_store_left_behind() {
    // Store a known i64 at offset 0, then return — and read it back via the Inspector.
    const MAGIC: u64 = 0x1122334455667788;
    let src = r#"
memory 17
func () -> (i32) {
block0():
  v0 = i64.const 0
  v1 = i64.const 1234605616436508552
  i64.store v0 v1
  v2 = i32.const 0
  return v2
}
"#;
    let m = parse_module(src).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[], 1_000_000);
    let _ = finished_ok(insp.run_until_stop());
    let bytes = insp.read_window(0, 8).expect("read window");
    assert_eq!(bytes, MAGIC.to_le_bytes());
}

// A store to a fixed window offset, then return — for write-watchpoint tests.
const STORE_PROG: &str = r#"
memory 17
func () -> (i32) {
block0():
  v0 = i64.const 0
  v1 = i64.const 1234605616436508552
  i64.store v0 v1
  v2 = i32.const 0
  return v2
}
"#;

#[test]
fn write_watchpoint_stops_before_the_store_then_step_applies_it() {
    const MAGIC: u64 = 0x1122334455667788;
    let m = parse_module(STORE_PROG).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[], 1_000_000);
    insp.set_watchpoint(0, 8, WatchKind::Write);

    // Pauses *before* the store; the watched bytes are still the initial zeros.
    match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Watchpoint { addr, write },
            pc,
        } => {
            assert_eq!((addr, write), (0, true));
            assert_eq!(
                pc,
                IrPc {
                    module: 0,
                    func: 0,
                    block: 0,
                    inst: 2
                }
            );
        }
        other => panic!("expected write watchpoint, got {other:?}"),
    }
    assert_eq!(insp.read_window(0, 8).unwrap(), [0u8; 8]);

    // One step applies the store; now the new bytes are visible.
    let _ = insp.step();
    assert_eq!(insp.read_window(0, 8).unwrap(), MAGIC.to_le_bytes());
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I32(0)]);
}

#[test]
fn write_watchpoint_does_not_fire_on_a_read_and_read_watch_does() {
    // Load from offset 0, then return it. A *write* watch must ignore the load; a *read* watch fires.
    let src = r#"
memory 17
func () -> (i64) {
block0():
  v0 = i64.const 0
  v1 = i64.load v0
  return v1
}
"#;
    let m = parse_module(src).expect("parse");

    // Write-kind watch: the load is not a write, so the guest runs clean to completion.
    let mut insp = Inspector::attach(&m, 0, &[], 1_000_000);
    insp.set_watchpoint(0, 8, WatchKind::Write);
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I64(0)]);

    // Read-kind watch on the same range: pauses before the load.
    let mut insp = Inspector::attach(&m, 0, &[], 1_000_000);
    insp.set_watchpoint(0, 8, WatchKind::Read);
    match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Watchpoint { addr, write },
            ..
        } => {
            assert_eq!((addr, write), (0, false));
        }
        other => panic!("expected read watchpoint, got {other:?}"),
    }
}

#[test]
fn clearing_a_watchpoint_lets_the_guest_run_clean() {
    let m = parse_module(STORE_PROG).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[], 1_000_000);
    let id = insp.set_watchpoint(0, 8, WatchKind::Write);
    assert!(insp.clear_watchpoint(id));
    assert!(!insp.clear_watchpoint(id), "second clear is a no-op");
    // With the watch gone, no pause: the store runs and the function returns 0.
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I32(0)]);
}

#[test]
fn watchpoint_outside_the_accessed_range_does_not_fire() {
    // Watch [64, 72): the store hits [0, 8), which does not overlap, so the guest runs clean.
    let m = parse_module(STORE_PROG).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[], 1_000_000);
    insp.set_watchpoint(64, 8, WatchKind::Write);
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I32(0)]);
}

// Writes "Hi" into the window then `cap.call 0 1` (Stream::write) of 2 bytes to a stdout handle
// passed as v0 — the standard capability-using shape (§3c/§3e).
const CAP_WRITE: &str = r#"
memory 16
func (i32) -> (i64) {
block0(v0: i32):
  v1 = i64.const 0
  v2 = i32.const 72
  i32.store8 v1 v2
  v3 = i64.const 1
  v4 = i32.const 105
  i32.store8 v3 v4
  v5 = i64.const 0
  v6 = i64.const 2
  v7 = cap.call 0 1 (i64, i64) -> (i64) v0 (v5, v6)
  return v7
}
"#;

#[test]
fn debugs_a_capability_using_guest_end_to_end() {
    // attach_with_host: grant a stdout stream, pass its handle as v0, run to completion, and read
    // the captured host-side effect back through the Inspector.
    let m = parse_module(CAP_WRITE).expect("parse");
    let mut host = Host::new();
    let stdout = host.grant_stream(StreamRole::Out);
    let mut insp = Inspector::attach_with_host(&m, 0, &[Value::I32(stdout)], 100_000, host);
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I64(2)]);
    assert_eq!(insp.host().stdout, b"Hi".to_vec());
}

#[test]
fn cap_call_stop_pauses_at_the_host_boundary_before_the_effect() {
    let m = parse_module(CAP_WRITE).expect("parse");
    let mut host = Host::new();
    let stdout = host.grant_stream(StreamRole::Out);
    let mut insp = Inspector::attach_with_host(&m, 0, &[Value::I32(stdout)], 100_000, host);
    insp.set_cap_call_stops(true);

    // Stops *before* the write (Stream = type_id 0, write = op 1); the handle is live as v0.
    match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::CapCall { type_id, op },
            ..
        } => {
            assert_eq!((type_id, op), (0, 1));
        }
        other => panic!("expected cap.call stop, got {other:?}"),
    }
    assert_eq!(insp.read_ir_value(0, 0), Some(Value::I32(stdout)));
    assert!(
        insp.host().stdout.is_empty(),
        "effect not applied before the boundary stop"
    );

    // Step performs the call: now the bytes are captured and the count is returned.
    let _ = insp.step();
    assert_eq!(insp.host().stdout, b"Hi".to_vec());
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I64(2)]);
}

#[test]
fn cap_call_stops_off_by_default() {
    // Without the toggle, a cap.call is just another op — no pause.
    let m = parse_module(CAP_WRITE).expect("parse");
    let mut host = Host::new();
    let stdout = host.grant_stream(StreamRole::Out);
    let mut insp = Inspector::attach_with_host(&m, 0, &[Value::I32(stdout)], 100_000, host);
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I64(2)]);
}

// ---- W4: the frontend-neutral debug-info waist (source locations + named variables) ----

// LOOP_SUM with a hand-written debug-info section: a source location at the loop body and the two
// loop variables mapped to their block-relative SSA value indices (i = v2, acc = v3 in block1).
const LOOP_SUM_DBG: &str = r#"
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

debug.file 0 "sum.c"
debug.loc 0 1 0 0 7 5
debug.var 0 "i" ssa 0 "int"
debug.var 0 "acc" ssa 1 "int"
"#;

#[test]
fn source_location_and_named_vars_at_a_breakpoint() {
    let m = parse_module(LOOP_SUM_DBG).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(3)], 1_000_000);
    let bp = IrPc {
        module: 0,
        func: 0,
        block: 1,
        inst: 0,
    };
    insp.set_breakpoint(bp);
    assert!(matches!(insp.run_until_stop(), Stop::Break { .. }));

    // The IR location resolves to source (sum.c:7:5).
    assert_eq!(
        insp.source_loc(bp),
        Some(SourceLoc {
            file: "sum.c".into(),
            line: 7,
            col: 5
        })
    );
    // The backtrace frame carries the source location too.
    assert_eq!(insp.backtrace()[0].source.as_ref().map(|s| s.line), Some(7));

    // Named source variables resolve to their live values: first iteration i = 3, acc = 0.
    assert_eq!(
        insp.read_var(0, "i", 4),
        Some(VarValue::Value(Value::I32(3)))
    );
    assert_eq!(
        insp.read_var(0, "acc", 4),
        Some(VarValue::Value(Value::I32(0)))
    );
    assert_eq!(insp.read_var(0, "nope", 4), None);
}

#[test]
fn debug_info_round_trips_through_text() {
    // Includes a window-located variable to exercise that VarLoc on the wire.
    let src = r#"
func () -> (i32) {
block0():
  v0 = i32.const 0
  return v0
}

debug.file 0 "a.c"
debug.loc 0 0 0 0 1 1
debug.var 0 "x" ssa 0 "int"
debug.var 0 "buf" win 16 "char"
"#;
    let m = parse_module(src).expect("parse");
    let di = m.debug_info.as_ref().expect("debug info present");
    assert_eq!(di.files, vec!["a.c".to_string()]);
    assert_eq!(di.locs.len(), 1);
    assert_eq!(di.vars.len(), 2);

    // parse → print → parse is stable (the debug section round-trips).
    let m2 = parse_module(&print_module(&m)).expect("reparse");
    assert_eq!(m, m2);
}

#[test]
fn no_debug_info_means_none_and_the_inspector_still_works() {
    let m = parse_module(LOOP_SUM).expect("parse");
    assert!(m.debug_info.is_none());
    let bp = IrPc {
        module: 0,
        func: 0,
        block: 1,
        inst: 0,
    };
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(2)], 1_000_000);
    insp.set_breakpoint(bp);
    let _ = insp.run_until_stop();
    assert_eq!(insp.source_loc(bp), None);
    assert_eq!(insp.read_var(0, "i", 4), None);
}

// --- W1 time-travel: seek / step_back via stateless re-execution (DEBUGGING.md W1) -------------

/// `seek(t)` re-executes the run to logical time `t` and restores the *exact* frame state there;
/// `step_back` walks the clock down. Verified against snapshots taken while stepping forward.
#[test]
fn seek_and_step_back_time_travel() {
    let m = parse_module(LOOP_SUM).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(3)], 100_000);

    // Full frame state (pc + live SSA values per frame) — the thing time-travel must reproduce.
    fn snap(i: &Inspector) -> Vec<(IrPc, Vec<Value>)> {
        i.backtrace()
            .iter()
            .map(|f| (f.pc, f.vals.clone()))
            .collect()
    }

    // Walk forward, recording (logical time → state) at each stop.
    let mut at: std::collections::BTreeMap<u64, Vec<(IrPc, Vec<Value>)>> = Default::default();
    at.insert(insp.clock(), snap(&insp)); // clock 0, before the first op
    for _ in 0..6 {
        match insp.step() {
            Stop::Break { .. } => {
                at.insert(insp.clock(), snap(&insp));
            }
            Stop::Finished(_) => break,
            Stop::Blocked => panic!("unexpected block"),
        }
    }
    assert!(at.len() >= 4, "stepped through several ops: {}", at.len());

    // Time-travel to each recorded instant (out of order) and verify exact restoration.
    for (&t, want) in at.iter().rev() {
        match insp.seek(t) {
            Stop::Break { .. } => {}
            other => panic!("seek({t}) should land at a paused op, got {other:?}"),
        }
        assert_eq!(insp.clock(), t, "seek({t}) lands at logical time {t}");
        assert_eq!(
            &snap(&insp),
            want,
            "seek({t}) restores the exact frame state"
        );
    }

    // step_back decrements the clock one op at a time.
    insp.seek(3);
    assert!(matches!(insp.step_back(), Stop::Break { .. }));
    assert_eq!(insp.clock(), 2, "step_back from clock 3 lands at 2");
    assert_eq!(snap(&insp), at[&2], "and restores that state");

    // After time-traveling, resuming forward still runs to the correct result (sum 1+2+3 = 6).
    insp.seek(0);
    match insp.run_until_stop() {
        Stop::Finished(Ok(vs)) => assert_eq!(vs, vec![Value::I32(6)]),
        other => panic!("expected Finished(6), got {other:?}"),
    }
}

/// Seeking past the end of the run lands on the finished result, not a pause.
#[test]
fn seek_past_the_end_finishes() {
    let m = parse_module(LOOP_SUM).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(2)], 100_000);
    match insp.seek(u64::MAX) {
        Stop::Finished(Ok(vs)) => assert_eq!(vs, vec![Value::I32(3)]), // 1+2
        other => panic!("expected Finished, got {other:?}"),
    }
    // And we can still seek back to the start afterward.
    assert!(matches!(insp.seek(0), Stop::Break { .. }));
    assert_eq!(insp.clock(), 0);
}

// --- W1 CapTape: record/replay the nondeterministic cap inputs so seek is faithful --------------

// Reads the Clock capability (iface 2, op 0 `now`) twice and returns their sum. The two reads are a
// nondeterministic *input*: their values depend on the host's clock, which a fresh re-execution
// cannot reproduce without the recorded tape.
const CLOCK_SUM: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = cap.call 2 0 () -> (i64) v0 ()
  v2 = cap.call 2 0 () -> (i64) v0 ()
  v3 = i64.add v1 v2
  return v3
}
"#;

#[test]
fn captape_replays_clock_inputs_for_faithful_seek() {
    let m = parse_module(CLOCK_SUM).expect("parse");
    let mut host = Host::new();
    host.clock_ns = 1000; // a nonzero clock a fresh powerbox would not reproduce
    let clk = host.grant_clock();
    let mut insp = Inspector::attach_with_host(&m, 0, &[Value::I32(clk)], 100_000, host);

    // Forward run: now()=1000 then 1001 → 2001.
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I64(2001)]);

    // The tape captured the two Clock crossings, in order, with their result slots.
    let tape = insp.cap_tape();
    assert_eq!(tape.records.len(), 2, "two Clock reads taped");
    assert_eq!(tape.records[0].result, Ok(vec![1000]));
    assert_eq!(tape.records[1].result, Ok(vec![1001]));

    // Time-travel to the start and replay forward: the CapTape feeds the recorded clock values, so
    // we reproduce 2001 — not the 0+1=1 a fresh empty powerbox would yield.
    insp.seek(0);
    assert_eq!(
        finished_ok(insp.run_until_stop()),
        vec![Value::I64(2001)],
        "seek replays the recorded nondeterministic inputs"
    );

    // Contrast: the same guest on a fresh clock (no tape) reads 0,1 → 1, proving the input really is
    // nondeterministic and that the tape (not luck) is what made the replay faithful.
    let mut h2 = Host::new(); // clock_ns defaults to 0
    let c2 = h2.grant_clock();
    let mut insp2 = Inspector::attach_with_host(&m, 0, &[Value::I32(c2)], 100_000, h2);
    assert_eq!(finished_ok(insp2.run_until_stop()), vec![Value::I64(1)]);
}

// Reads 2 bytes from a stdin stream (iface 0, op 0 `read`) into window[0..2], then returns the
// first byte. The bytes are a nondeterministic input that crosses via *guest memory* (the read
// fills the buffer), so faithful replay needs the captured buffer write, not just a result slot.
const STDIN_FIRST: &str = r#"
memory 16
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 0
  v2 = i64.const 2
  v3 = cap.call 0 0 (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i64.const 0
  v5 = i32.load8_u v4
  return v5
}
"#;

#[test]
fn captape_replays_stdin_read_into_the_buffer_for_faithful_seek() {
    let m = parse_module(STDIN_FIRST).expect("parse");
    let mut host = Host::new();
    host.stdin = b"Hi".to_vec();
    let stdin = host.grant_stream(StreamRole::In);
    let mut insp = Inspector::attach_with_host(&m, 0, &[Value::I32(stdin)], 100_000, host);

    // Forward: read "Hi" into the buffer, return buf[0] = 'H' = 72.
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I32(72)]);

    // The tape captured the read, including the bytes it wrote into the guest window.
    let tape = insp.cap_tape();
    assert_eq!(tape.records.len(), 1);
    assert_eq!(tape.records[0].result, Ok(vec![2])); // 2 bytes read
    assert_eq!(tape.records[0].mem_writes, vec![(0u64, b"Hi".to_vec())]);

    // Time-travel to the start and replay: the read's buffer write is re-applied from the tape, so
    // we reproduce 72 — even though the replay host has empty stdin.
    insp.seek(0);
    assert_eq!(
        finished_ok(insp.run_until_stop()),
        vec![Value::I32(72)],
        "stdin read replays its buffer bytes on seek"
    );

    // Contrast: a fresh host with empty stdin reads 0 bytes, so buf[0] stays 0.
    let mut h2 = Host::new();
    let s2 = h2.grant_stream(StreamRole::In);
    let mut insp2 = Inspector::attach_with_host(&m, 0, &[Value::I32(s2)], 100_000, h2);
    assert_eq!(finished_ok(insp2.run_until_stop()), vec![Value::I32(0)]);
}

// Calls a host-fn capability (iface 13) twice and sums the results — the general embedder
// nondeterminism escape hatch. The closure is gone on a fresh powerbox, so ONLY the CapTape can
// reproduce its outputs across time-travel.
const HOSTFN_SUM: &str = r#"
func (i32) -> (i64) {
block0(v0: i32):
  v1 = cap.call 13 0 () -> (i64) v0 ()
  v2 = cap.call 13 0 () -> (i64) v0 ()
  v3 = i64.add v1 v2
  return v3
}
"#;

#[test]
fn captape_replays_host_fn_inputs_for_faithful_seek() {
    let m = parse_module(HOSTFN_SUM).expect("parse");
    let mut host = Host::new();
    // A nondeterministic host capability: each call returns an incrementing counter.
    let mut n = 100i64;
    let hf = host.grant_host_fn(Box::new(move |_op, _args, _mem| {
        let v = n;
        n += 1;
        Ok(vec![v])
    }));
    let mut insp = Inspector::attach_with_host(&m, 0, &[Value::I32(hf)], 100_000, host);

    // Forward: 100 + 101 = 201; both crossings taped.
    assert_eq!(finished_ok(insp.run_until_stop()), vec![Value::I64(201)]);
    assert_eq!(insp.cap_tape().records.len(), 2);
    assert_eq!(insp.cap_tape().records[0].result, Ok(vec![100]));

    // Time-travel: the host-fn closure does not exist on the fresh replay powerbox, so without the
    // tape the re-run would fault. seek(0)+resume still yields 201 — the tape carried the outputs.
    insp.seek(0);
    assert_eq!(
        finished_ok(insp.run_until_stop()),
        vec![Value::I64(201)],
        "host-fn outputs replay from the tape, not a (vanished) live closure"
    );
}
