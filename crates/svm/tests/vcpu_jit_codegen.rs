//! Browser §22 **real-codegen** slice — native proof of the `JitInvoke` external-result seam on the
//! resumable `Vcpu`.
//!
//! Today the browser services a guest's `Jit.invoke` by running the submitted unit on the **bytecode
//! interpreter** ([`Vcpu::deliver_jit_invoke`], which `compile_module`s the unit's IR and `run_invoke`s
//! it). The real-codegen tier instead **emits wasm** for the unit and runs `f{entry}(win, env, args)`
//! on the Worker, then delivers the emitted region's results back via
//! [`Vcpu::deliver_jit_invoke_vals`]. This test stands in for that host **without any wasm**: it
//! services each `JitInvoke` by computing the unit on a standalone bytecode run (exactly what the
//! emitted region computes) and delivering the i64 result slots — then asserts the whole-vCPU result
//! is **identical** to servicing the same event the interpreter way. So the codegen marshalling (args
//! in, results out, resume) is exact, the same contract `vcpu_tierup.rs` holds for the tier-up seam.

use std::sync::{Arc, Mutex};
use svm_interp::{bytecode, Host, Region, Trap, Value};
use svm_run::grant_jit;
use svm_text::parse_module;
use svm_verify::verify_module;

/// The pre-compiled unit the guest invokes: `unit(a, b) = a*3 + b` — pure i64 compute, no host/memory
/// use (the shape the browser emits + calls as `f0(win, env, a, b)`).
const UNIT: &str = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  v3 = i64.const 3
  vm = i64.mul va v3
  vr = i64.add vm vb
  return vr
  }
}
"#;

/// A unit that **traps**: `unit(x) = 100 / (x - 7)` — div-by-zero at `x == 7`. Proves a codegen-run
/// trap surfaces exactly where the interpreter's invoke would ([`Vcpu::deliver_jit_invoke_trap`]).
const UNIT_TRAP: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vx: i64) {
  v7 = i64.const 7
  vd = i64.sub vx v7
  v100 = i64.const 100
  vr = i64.div_s v100 vd
  return vr
  }
}
"#;

/// Guest `(jit, code) -> (i64)`: invoke the unit with `(a, b)` and return its result. Single vCPU —
/// no threads; the invoke is the only host event.
const GUEST: &str = r#"memory 16
func (i32, i32) -> (i64) {
block 0 (vjit: i32, vcode: i32) {
  vc = i64.extend_i32_u vcode
  va = i64.const 4
  vb = i64.const 5
  vr = cap.call 11 1 (i64, i64, i64) -> (i64) vjit (vc, va, vb)
  return vr
  }
}
"#;

/// Guest `(jit, code) -> (i64)`: invoke the trapping unit with `x = 7`.
const GUEST_TRAP: &str = r#"memory 16
func (i32, i32) -> (i64) {
block 0 (vjit: i32, vcode: i32) {
  vc = i64.extend_i32_u vcode
  vx = i64.const 7
  vr = cap.call 11 1 (i64, i64) -> (i64) vjit (vc, vx)
  return vr
  }
}
"#;

/// A fresh 64 KiB window for a root vCPU. Leaked for the test's lifetime.
fn window() -> Arc<Region> {
    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero 8-aligned layout; leaked for the process — never freed, so no aliasing.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `size` valid 8-aligned bytes, owned here and never freed.
    Arc::new(unsafe { Region::shared(base, size as u64) })
}

/// A powerbox granted the `Jit` cap with `unit_src` host-compiled into it; returns `(host, jit, code)`.
fn powerbox_with_unit(guest: &svm_ir::Module, unit_src: &str) -> (Host, i32, i32) {
    let mut host = Host::new();
    let jit = grant_jit(&mut host, guest, 4); // sets the blob validator; 2^4 = 16-slot table
    let unit = {
        let m = parse_module(unit_src).expect("parse unit");
        verify_module(&m).expect("verify unit");
        svm_encode::encode_module(&m)
    };
    let code = host
        .jit_compile(jit, &unit)
        .expect("no trap")
        .expect("compile ok")
        .handle;
    (host, jit, code)
}

/// How a `JitInvoke` is serviced.
#[derive(Clone, Copy)]
enum Mode {
    /// The engine runs the unit on the bytecode interpreter (`deliver_jit_invoke`) — today's path.
    Interp,
    /// The host runs the unit externally (here: a standalone bytecode run, standing in for emitted
    /// wasm) and delivers the i64 result slots (`deliver_jit_invoke_vals`) — the real-codegen seam.
    Codegen,
}

/// Drive the single-vCPU guest to completion, servicing its one `JitInvoke` in `mode`. The powerbox's
/// `resolve_unit` mirrors the browser host (authority + cross-domain check → the unit's funcs).
fn run(guest_src: &str, unit_src: &str, mode: Mode) -> Result<Vec<Value>, Trap> {
    let m = parse_module(guest_src).unwrap();
    verify_module(&m).expect("verify guest");
    let unit_m = parse_module(unit_src).unwrap();
    let (host, jit, code) = powerbox_with_unit(&m, unit_src);
    let pb = Mutex::new(host);
    let prog = bytecode::VcpuProgram::compile_with_jit_table(&m, 4).unwrap();
    let back = window();
    let mut vcpu =
        bytecode::Vcpu::new_root(&prog, 0, &[Value::I32(jit), Value::I32(code)], back, &[])
            .expect("root");

    let resolve_unit = |handle: i32, code: i32| -> Result<Arc<[svm_ir::Func]>, Trap> {
        let g = pb.lock().unwrap();
        let domain = g.resolve_jit_domain(handle)?;
        let (cd, cu) = g.resolve_jit_code(code)?;
        if cd != domain {
            return Err(Trap::CapFault);
        }
        g.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
    };

    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => return Ok(vals),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::JitInvoke {
                handle,
                code,
                argv,
                params: _,
                results,
            } => match mode {
                Mode::Interp => vcpu.deliver_jit_invoke(resolve_unit(handle, code)),
                Mode::Codegen => {
                    // Authority still resolves through the powerbox (a forged handle must trap
                    // identically); then run the unit standalone over argv — what emitted `f0` computes.
                    match resolve_unit(handle, code) {
                        Err(t) => vcpu.deliver_jit_invoke_trap(t),
                        Ok(_funcs) => {
                            let args: Vec<Value> = argv.iter().map(|&s| Value::I64(s)).collect();
                            let mut fuel = u64::MAX;
                            match bytecode::compile_and_run(&unit_m, 0, &args, &mut fuel)
                                .expect("unit supported")
                            {
                                Ok(vals) => {
                                    let slots: Vec<i64> = vals
                                        .iter()
                                        .zip(results.iter())
                                        .map(|(v, _)| match v {
                                            Value::I64(x) => *x,
                                            Value::I32(x) => *x as i64,
                                            _ => panic!("non-integer unit result"),
                                        })
                                        .collect();
                                    vcpu.deliver_jit_invoke_vals(&slots);
                                }
                                Err(t) => vcpu.deliver_jit_invoke_trap(t),
                            }
                        }
                    }
                }
            },
            _ => panic!("unexpected event (this guest only invokes)"),
        }
    }
}

#[test]
fn codegen_invoke_matches_interp() {
    // unit(4, 5) = 4*3 + 5 = 17, both ways.
    let want = run(GUEST, UNIT, Mode::Interp);
    let got = run(GUEST, UNIT, Mode::Codegen);
    assert_eq!(want, Ok(vec![Value::I64(17)]), "interp invoke value");
    assert_eq!(got, want, "codegen invoke diverged from interp invoke");
}

#[test]
fn codegen_invoke_trap_parity() {
    // unit(7) = 100/(7-7) → div-by-zero; the codegen seam must trap iff the interp invoke does.
    let want = run(GUEST_TRAP, UNIT_TRAP, Mode::Interp);
    let got = run(GUEST_TRAP, UNIT_TRAP, Mode::Codegen);
    assert!(want.is_err(), "interp invoke must trap at x=7");
    assert_eq!(
        want.is_err(),
        got.is_err(),
        "codegen invoke trap parity broke (interp err={:?})",
        want.is_err()
    );
}
