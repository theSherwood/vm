//! ISSUES.md I41 — **graceful revocation**: a `cap.call` through a handle that was once granted
//! and has since been revoked completes with the probeable `-EBADF` errno the §3.6
//! revocation-unpark already delivers ("cancellation is a value", whether the caller was parked
//! mid-call or calls a moment later). The D37 trap stays reserved for what it always meant:
//! a **forgery** (a generation the slot never issued), and type-confusion on a **live** handle.
//! No tombstone storage: a slot's generation advances only at (re)grant, so every generation
//! `1..=current` was once live — a dead-but-issued generation IS the tombstone.
//!
//! Differential across all three backends — tree-walk, bytecode (the module compiles natively),
//! and the JIT (`Clock.now` additionally exercises the D45 fast-cap path, which must answer
//! byte-identically to the generic dispatch).

use svm_interp::{bytecode, run_with_host, run_with_host_fast, Host, StreamRole, Trap, Value};
use svm_ir::DEFAULT_RESERVED_LOG2;
use svm_jit::{JitOutcome, TrapKind};
use svm_run::jit_cap_run;
use svm_text::parse_module;
use svm_verify::verify_module;

/// `main(stream_h, clock_h)`: write through the stream handle, read the clock, and pack both
/// results as `write * 1000 + now` — through revoked handles that is `-9 * 1000 + -9 = -9009`.
const CALLER: &str = r#"
memory 12

func (i32, i32) -> (i64) {
block 0 (vs: i32, vc: i32) {
  vp = i64.const 0
  vl = i64.const 0
  vw = cap.call 0 1 (i64, i64) -> (i64) vs (vp, vl)
  vn = cap.call 2 0 () -> (i64) vc ()
  vk = i64.const 1000
  vm = i64.mul vw vk
  vr = i64.add vm vn
  return vr
  }
}
"#;

/// A **forged** handle — slot 200 with generation 5, never granted by any test host (a fresh
/// powerbox grants two handles into slots 0/1) — must still trap: `(5 << 8) | 200 = 1480`.
const FORGER: &str = r#"
memory 12

func () -> (i64) {
block 0 () {
  vh = i32.const 1480
  vp = i64.const 0
  vl = i64.const 0
  vw = cap.call 0 1 (i64, i64) -> (i64) vh (vp, vl)
  return vw
  }
}
"#;

fn module(src: &str) -> svm_ir::Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

/// A host with a granted-then-revoked stdout stream and clock; returns `(host, hs, hc)`.
fn revoked_host() -> (Host, i32, i32) {
    let mut host = Host::new();
    let hs = host.grant_stream(StreamRole::Out);
    let hc = host.grant_clock();
    host.close(hs);
    host.close(hc);
    (host, hs, hc)
}

/// A revoked-once-valid handle answers `-EBADF` on every backend — never a trap.
#[test]
fn a_revoked_handle_completes_with_an_errno_on_all_backends() {
    let m = module(CALLER);
    assert!(
        bytecode::compile_module(&m.funcs).is_some(),
        "the caller must run natively on the bytecode engine (a real 3-way differential)"
    );
    let expect = vec![Value::I64(-9 * 1000 + -9)];

    let (mut hi, hs, hc) = revoked_host();
    let mut fuel = 1_000_000u64;
    let args = [Value::I32(hs), Value::I32(hc)];
    assert_eq!(
        run_with_host(&m, 0, &args, &mut fuel, &mut hi),
        Ok(expect.clone())
    );

    let (mut hb, hs, hc) = revoked_host();
    let mut fuel = 1_000_000u64;
    let args = [Value::I32(hs), Value::I32(hc)];
    assert_eq!(
        run_with_host_fast(&m, 0, &args, &mut fuel, &mut hb),
        Ok(expect)
    );

    let (mut hj, hs, hc) = revoked_host();
    let (jout, _) = jit_cap_run(
        &m,
        0,
        &[hs as i64, hc as i64],
        &[],
        DEFAULT_RESERVED_LOG2,
        0,
        &mut hj,
    )
    .expect("jit run");
    assert_eq!(
        jout,
        JitOutcome::Returned(vec![-9009]),
        "the JIT (incl. the Clock fast-cap path) answers the same errno"
    );
}

/// A forged handle — a generation the slot never issued — still traps on every backend (D37).
#[test]
fn a_forged_handle_still_traps_on_all_backends() {
    let m = module(FORGER);
    assert!(bytecode::compile_module(&m.funcs).is_some());

    let mut hi = Host::new();
    hi.grant_stream(StreamRole::Out);
    hi.grant_clock();
    let mut fuel = 1_000_000u64;
    assert_eq!(
        run_with_host(&m, 0, &[], &mut fuel, &mut hi),
        Err(Trap::CapFault)
    );

    let mut hb = Host::new();
    hb.grant_stream(StreamRole::Out);
    hb.grant_clock();
    let mut fuel = 1_000_000u64;
    assert_eq!(
        run_with_host_fast(&m, 0, &[], &mut fuel, &mut hb),
        Err(Trap::CapFault)
    );

    let mut hj = Host::new();
    hj.grant_stream(StreamRole::Out);
    hj.grant_clock();
    let (jout, _) =
        jit_cap_run(&m, 0, &[], &[], DEFAULT_RESERVED_LOG2, 0, &mut hj).expect("jit run");
    assert_eq!(jout, JitOutcome::Trapped(TrapKind::CapFault));
}

/// Type confusion on a **live** handle stays a trap: revocation softened exactly one failure
/// (the once-valid dead handle), not the typing discipline.
#[test]
fn a_wrong_type_use_of_a_live_handle_still_traps() {
    // The caller writes through its FIRST arg as a stream — hand it the (live) clock handle.
    let m = module(CALLER);
    let mut hi = Host::new();
    let _hs = hi.grant_stream(StreamRole::Out);
    let hc = hi.grant_clock();
    let mut fuel = 1_000_000u64;
    assert_eq!(
        run_with_host(&m, 0, &[Value::I32(hc), Value::I32(hc)], &mut fuel, &mut hi),
        Err(Trap::CapFault)
    );
}
