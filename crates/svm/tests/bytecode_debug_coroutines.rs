//! Debugging **§14 coroutines** (`Instantiator.spawn_coroutine` / `resume` + `Yielder.yield`) on the
//! single-vCPU bytecode `DebugRun`. A coroutine is a cooperative confined child driven **inline** by the
//! debug engine (never via the thread scheduler), exactly as the production engine drives it — so a
//! `spawn_coroutine`/`resume`/`yield` round-trip runs correctly under the debugger, breakpoints fire in
//! the coroutine-using **parent**, and the whole run stays bit-identical to the production bytecode
//! engine + tree-walker oracle. (The coroutine body itself is stepped opaquely by `resume_coro` here;
//! step-*into* the body is a follow-up.)

use svm_interp::bytecode::{self, DebugRun};
use svm_interp::{run_with_host, Host, IrPc, Value};
use svm_text::parse_module;

// Same fixture as `bytecode_coroutines.rs`: the parent (func 0) spawns a coroutine confined to
// `[64 KiB, 128 KiB)` and resumes it three times; the child (func 1) yields 100, then 200+r1, then
// returns 999+r2, where r1/r2 are the values the parent delivers (10, 20). The Instantiator handle
// reaches the guest as func 0's argument. Result: 100 + 210 + 1019 + RETURNED*1_000_000 = 1_001_329.
const CORO: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i64.const 10
  v10, v11 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v9)
  v12 = i64.const 20
  v13, v14 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v12)
  v15 = i64.add v8 v11
  v16 = i64.add v15 v14
  v17 = i64.extend_i32_s v13
  v18 = i64.const 1000000
  v19 = i64.mul v17 v18
  v20 = i64.add v16 v19
  return v20
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 7
  i32.store8 v2 v3
  v4 = i64.const 100
  v5 = cap.call 7 0 (i64) -> (i64) v1 (v4)
  v6 = i64.const 200
  v7 = i64.add v6 v5
  v8 = cap.call 7 0 (i64) -> (i64) v1 (v7)
  v9 = i64.const 999
  v10 = i64.add v9 v8
  return v10
  }
}
"#;

const WANT: i64 = 100 + 210 + 1019 + 1_000_000;

/// A `DebugRun` carrying a host with a granted §14 Instantiator, entered at func 0 with the cap handle
/// as its argument — the debug analogue of the coroutine equality harness's setup.
fn coro_session() -> DebugRun {
    let m = parse_module(CORO).expect("parse");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    DebugRun::new_with_host(&m, 0, &[Value::I32(inst)], host)
        .expect("bytecode debug engine must drive §14 coroutines")
}

/// Drive a `DebugRun` to completion, counting stops at `bps`, returning the result.
fn drive(run: &mut DebugRun, bps: &[IrPc], fuel: &mut u64) -> (usize, Result<Vec<Value>, ()>) {
    let mut hits = 0;
    loop {
        match run.run_to(bps, fuel) {
            Some(_) => hits += 1,
            None => return (hits, run.result().cloned().unwrap().map_err(|_| ())),
        }
    }
}

/// The full coroutine run through the debugger matches the production bytecode engine **and** the
/// tree-walker oracle — the inline `spawn_coroutine`/`resume`/`yield` handoffs don't perturb the result
/// (this whole module was `Malformed`/declined under the debugger before this slice).
#[test]
fn coroutine_debug_run_matches_the_oracle() {
    let mut r = coro_session();
    let mut fuel = 5_000_000u64;
    let (_, res) = drive(&mut r, &[], &mut fuel);
    assert_eq!(
        res,
        Ok(vec![Value::I64(WANT)]),
        "100 + 210 + 1019 + RETURNED*1e6"
    );

    let m = parse_module(CORO).unwrap();
    let mut h_bc = Host::new();
    let inst_bc = h_bc.grant_instantiator(0, 128 << 10);
    let mut f_bc = 5_000_000u64;
    let bc =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(inst_bc)], &mut f_bc, &mut h_bc)
            .expect("bytecode engine drives coroutines");
    assert_eq!(
        bc.clone().map_err(|_| ()),
        res,
        "debug run ≡ production bytecode"
    );

    let mut h_tw = Host::new();
    let inst_tw = h_tw.grant_instantiator(0, 128 << 10);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &[Value::I32(inst_tw)], &mut f_tw, &mut h_tw);
    assert_eq!(tw, bc, "bytecode ≡ tree-walker");
}

/// The first `resume` in the coroutine parent (func 0, block 0, inst 6). A breakpoint here proves the
/// debugger drives a coroutine-using parent up to — and through — the cooperative handoff.
fn first_resume() -> IrPc {
    IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: 6,
    }
}

/// A breakpoint in the coroutine **parent** fires, and continuing past it drives the coroutine to the
/// correct result — the debugger follows the parent across the inline `resume` handoffs.
#[test]
fn breakpoint_in_a_coroutine_parent_fires() {
    let mut r = coro_session();
    let mut fuel = 5_000_000u64;
    assert_eq!(r.run_to(&[first_resume()], &mut fuel), Some(first_resume()));
    assert_eq!(
        r.frame_pc(0),
        Some(first_resume()),
        "stopped at the parent's first resume"
    );
    let (_, res) = drive(&mut r, &[], &mut fuel);
    assert_eq!(
        res,
        Ok(vec![Value::I64(WANT)]),
        "continues to the coroutine result"
    );
}

// A §14 **demand** coroutine (op 4): its window starts unmapped, so the child's first access is a
// recoverable fault that suspends to the parent (status FAULTED, value = fault address); the parent
// supplies `123` there and resumes, and the child's rewound load reads it and RETURNs. This drives the
// `CoStop::Fault` (supply-page-and-rewind) arm of the debug engine's inline `resume`. Same fixture as
// `bytecode_demand_coroutine.rs::DEMAND_SAME`. Result: 123 + FAULTED*1e6 + RETURNED*1e3.
const DEMAND_SAME: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 4 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i32.const 123
  i32.store8 v8 v9
  v10 = i64.const 0
  v11, v12 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v10)
  v13 = i64.extend_i32_s v7
  v14 = i64.const 1000000
  v15 = i64.mul v13 v14
  v16 = i64.extend_i32_s v11
  v17 = i64.const 1000
  v18 = i64.mul v16 v17
  v19 = i64.add v12 v15
  v20 = i64.add v19 v18
  return v20
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  return v3
  }
}
"#;

/// A **demand** coroutine faults, the parent supplies the page, and the rewound access re-reads it —
/// all under the debugger, matching the oracle. Exercises the recoverable-fault (`CoStop::Fault`) arm
/// of the debug engine's inline `resume`.
#[test]
fn demand_coroutine_debug_run_matches_the_oracle() {
    let m = parse_module(DEMAND_SAME).expect("parse");
    let (faulted, returned) = (2i64, 1i64);
    let want = 123 + faulted * 1_000_000 + returned * 1000; // 123 + FAULTED*1e6 + RETURNED*1e3

    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mut r = DebugRun::new_with_host(&m, 0, &[Value::I32(inst)], host)
        .expect("debug engine drives demand coroutines");
    let mut fuel = 5_000_000u64;
    let (_, res) = drive(&mut r, &[], &mut fuel);
    assert_eq!(res, Ok(vec![Value::I64(want)]), "demand coroutine result");

    let mut h_tw = Host::new();
    let inst_tw = h_tw.grant_instantiator(0, 128 << 10);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &[Value::I32(inst_tw)], &mut f_tw, &mut h_tw);
    assert_eq!(
        tw.map_err(|_| ()),
        res,
        "debug run ≡ tree-walker across the fault"
    );
}

/// Reverse debugging composes with coroutines: a fresh session ticked to an op clock reproduces the
/// exact parent position a forward run reached — what `seek` relies on across the inline handoffs.
#[test]
fn coroutine_tick_replays_deterministically() {
    let mut a = coro_session();
    let mut fuel = 5_000_000u64;
    assert_eq!(a.run_to(&[first_resume()], &mut fuel), Some(first_resume()));
    let clock = a.op_clock();

    let mut b = coro_session();
    let mut f2 = 5_000_000u64;
    while b.op_clock() < clock && b.tick(&mut f2) {}
    assert_eq!(b.op_clock(), clock, "replayed to the same op clock");
    assert_eq!(b.frame_pc(0), Some(first_resume()), "same parent position");
}
