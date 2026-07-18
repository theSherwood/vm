//! Debugging **§12 fibers** (`cont.new` / `cont.resume` / `suspend`) on the single-vCPU bytecode
//! `DebugRun`. A `cont.resume` switches the debugged continuation into the fiber, so breakpoints fire
//! inside the fiber body, the backtrace shows the fiber's stack, stepping descends into it, and the
//! whole run stays bit-identical to the production engine + tree-walker oracle across the switches.

use svm_interp::bytecode::{self, DebugRun};
use svm_interp::{run, IrPc, Value};
use svm_text::parse_module;

// A generator fiber: resume(10) suspends with 11 (SUSPENDED); resume(20) delivers 20 as the suspend
// result, the fiber returns 20+5=25 (RETURNED); the root sums 11+25 = 36. func 0 is the root, func 1
// the fiber: inst0 v0=const 1, inst1 v1=add varg v0, inst2 v2=suspend v1, inst3 v3=const 5,
// inst4 v4=add v2 v3, then return. Same fixture as `bytecode_fibers.rs`.
const SUSPEND_ROUNDTRIP: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 10
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 20
  v7, v8 = cont.resume v2 v6
  v9 = i64.add v5 v8
  return v9
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = i64.const 1
  v1 = i64.add varg v0
  v2 = suspend v1
  v3 = i64.const 5
  v4 = i64.add v2 v3
  return v4
}
"#;

/// Drive a `DebugRun` to completion, stopping only at `bps` (which it counts), returning the result.
fn drive(run: &mut DebugRun, bps: &[IrPc], fuel: &mut u64) -> (usize, Result<Vec<Value>, ()>) {
    let mut hits = 0;
    loop {
        match run.run_to(bps, fuel) {
            Some(_) => hits += 1,
            None => {
                return (hits, run.result().cloned().unwrap().map_err(|_| ()));
            }
        }
    }
}

/// A breakpoint **inside the fiber body** fires — proving `cont.resume` switched the debugged
/// continuation into the fiber and the debugger followed it — with the backtrace at the fiber's op.
#[test]
fn breakpoint_fires_inside_a_fiber() {
    let m = parse_module(SUSPEND_ROUNDTRIP).unwrap();
    let mut r = DebugRun::new(&m, 0, &[]).unwrap();
    let mut fuel = 1_000_000u64;
    let in_fiber = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 1,
    }; // `v1 = i64.add varg v0` in the fiber

    // The root does cont.new + cont.resume; the debugger follows into the fiber and stops there.
    assert_eq!(r.run_to(&[in_fiber], &mut fuel), Some(in_fiber));
    assert_eq!(
        r.frame_pc(0),
        Some(in_fiber),
        "the active continuation is the fiber's frame"
    );

    // Continue to completion: the breakpoint fires only once (the fiber runs its prefix once, then
    // resumes past the suspend), and the run produces the same result as the production engine.
    let (more, res) = drive(&mut r, &[in_fiber], &mut fuel);
    assert_eq!(
        more, 0,
        "the fiber prefix runs once — no re-hit after the suspend"
    );
    assert_eq!(res, Ok(vec![Value::I64(36)]), "11 + 25 = 36");
}

/// Stepping from the root's `cont.resume` **descends into the fiber** — its first op becomes the
/// active continuation, the fiber-debug analogue of stepping into a call.
#[test]
fn step_descends_into_a_resumed_fiber() {
    let m = parse_module(SUSPEND_ROUNDTRIP).unwrap();
    let mut r = DebugRun::new(&m, 0, &[]).unwrap();
    let mut fuel = 1_000_000u64;
    let resume = IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: 4,
    }; // `v4, v5 = cont.resume v2 v3` in the root

    assert_eq!(r.run_to(&[resume], &mut fuel), Some(resume));
    // Step over the resume: it switches into the fiber, so the next stop is the fiber's first op.
    match r.step(&mut fuel) {
        Some(pc) => assert_eq!(
            pc.func, 1,
            "stepped into the fiber (func 1), not past the resume"
        ),
        None => panic!("step ran to completion unexpectedly"),
    }
}

/// The full fiber run through the debugger matches the production bytecode engine **and** the
/// tree-walker oracle — the switches (`cont.new`/`resume`/`suspend`/return) don't perturb the result.
#[test]
fn fiber_debug_run_matches_the_oracle() {
    let m = parse_module(SUSPEND_ROUNDTRIP).unwrap();
    let mut r = DebugRun::new(&m, 0, &[]).unwrap();
    let mut fuel = 1_000_000u64;
    let (_, res) = drive(&mut r, &[], &mut fuel);
    assert_eq!(res, Ok(vec![Value::I64(36)]));

    let mut f_bc = 1_000_000u64;
    let bc =
        bytecode::compile_and_run(&m, 0, &[], &mut f_bc).expect("bytecode engine drives fibers");
    assert_eq!(
        bc.clone().map_err(|_| ()),
        res,
        "debug run ≡ production bytecode"
    );
    let mut f_tw = 1_000_000u64;
    assert_eq!(run(&m, 0, &[], &mut f_tw), bc, "bytecode ≡ tree-walker");
}

/// Reverse debugging composes with fibers: a fresh session ticked to an op clock reproduces the exact
/// position (including which fiber is active) a forward run reached — what `seek` relies on.
#[test]
fn fiber_tick_replays_deterministically() {
    let m = parse_module(SUSPEND_ROUNDTRIP).unwrap();
    let in_fiber = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 1,
    };
    // Forward to the in-fiber breakpoint; record the op clock + position.
    let mut a = DebugRun::new(&m, 0, &[]).unwrap();
    let mut fuel = 1_000_000u64;
    assert_eq!(a.run_to(&[in_fiber], &mut fuel), Some(in_fiber));
    let clock = a.op_clock();

    // A fresh run raw-ticked to that clock lands at the identical (fiber) position.
    let mut b = DebugRun::new(&m, 0, &[]).unwrap();
    let mut f2 = 1_000_000u64;
    while b.op_clock() < clock && b.tick(&mut f2) {}
    assert_eq!(b.op_clock(), clock, "replayed to the same op clock");
    assert_eq!(b.frame_pc(0), Some(in_fiber), "same fiber position");
}
