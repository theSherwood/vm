//! THREADS.md 4c-domain §14-D1 — §14 `spawn_coroutine_module` via the **resumable `Vcpu`** API. Unlike
//! §22 JIT (whose ops hand the raw cap handle to the orchestrating host to resolve), §14 resolves its
//! `Instantiator` authority **in-Vm** during `resume`, so the resumable vCPU must carry a real powerbox
//! ([`Vcpu::new_root_with_powerbox`]) rather than the deny-all default. With the grant in its own host,
//! `spawn_coroutine_module` is serviced **internally** inside [`Vcpu::run`] (resolve the granted module,
//! build the inline `Coro`) — the host never sees a §14 event, and the coroutine then runs inline via
//! `resume`. (A coroutine is single-vCPU by construction, so there is no thread orchestration here; the
//! `instantiate`/`instantiate_module` executor children — which need scheduler-driven confined child
//! vCPUs — are a separate slice.)
//!
//! The differential is vs the cooperative oracle (`compile_and_run_with_host`), with the same powerbox.
//! The `unsafe` (borrowing host memory via `Region::shared`) lives in this embedder/test.

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, Trap, Value};
use svm_text::parse_module;

/// The granted "plugin" module a coroutine runs: a 4 KiB window (`memory 12`) with a data segment
/// `"K"` (75) at offset 0; its entry `(i64 yielder) -> (i64)` returns that own data byte (75) — so its
/// result is deterministic (schedule-independent) and proves data-segment materialization.
const MODULE_CHILD: &str = r#"memory 12
data 0 "K"
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 0
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  return v3
}
"#;

/// Root `(instantiator, module) -> sum`: `spawn_coroutine_module` the granted module 8 times (one
/// coroutine per 4 KiB carve at `64 KiB + i*4 KiB`), `resume` each to completion, and sum the returned
/// values. Each module coroutine returns 75 ⇒ 8 × 75 = 600.
const COROUTINE_MODULE: &str = r#"memory 17
func (i32, i32) -> (i64) {
block0(vinst0: i32, vmod0: i32):
  vmod64 = i64.extend_i32_s vmod0
  vi0 = i64.const 0
  vs0 = i64.const 0
  br block1(vi0, vs0, vinst0, vmod64)
block1(vi: i64, vs: i64, vinst: i32, vmod: i64):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vs, vinst, vmod) block3(vs)
block2(vi2: i64, vs2: i64, vinst2: i32, vmod2: i64):
  v4096 = i64.const 4096
  vofflo = i64.mul vi2 v4096
  v64k = i64.const 65536
  voff = i64.add v64k vofflo
  ventry = i64.const 0
  vslog = i64.const 12
  vcoro = cap.call 6 6 (i64, i64, i64, i64) -> (i32) vinst2 (vmod2, ventry, voff, vslog)
  vrv = i64.const 0
  vstatus, vval = cap.call 6 3 (i32, i64) -> (i32, i64) vinst2 (vcoro, vrv)
  vsnew = i64.add vs2 vval
  v1 = i64.const 1
  vinext = i64.add vi2 v1
  br block1(vinext, vsnew, vinst2, vmod2)
block3(vs3: i64):
  return vs3
}
"#;

/// A fresh powerbox: `Instantiator` over the whole 128 KiB window + a `Module` grant for
/// [`MODULE_CHILD`]. Deterministic, so the resumable run and the oracle get identical handles.
fn powerbox() -> (Host, i32, i32) {
    let child = parse_module(MODULE_CHILD).expect("parse module child");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mh = host.grant_module(&child);
    (host, inst, mh)
}

/// Cooperative oracle: the same kernel + powerbox on the single-threaded engine.
fn run_cooperative() -> Result<Vec<Value>, Trap> {
    let m = parse_module(COROUTINE_MODULE).unwrap();
    let (mut host, inst, mh) = powerbox();
    let mut f = 50_000_000u64;
    bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(inst), Value::I32(mh)],
        &mut f,
        &mut host,
    )
    .expect("bytecode engine drives §14 spawn_coroutine_module (cooperative)")
}

/// Resumable path: a root `Vcpu` carrying the powerbox drives the kernel — every `spawn_coroutine_module`
/// is serviced internally, so the run finishes without the host servicing any event.
fn run_resumable() -> Result<Vec<Value>, Trap> {
    let m = parse_module(COROUTINE_MODULE).unwrap();
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let (host, inst, mh) = powerbox();

    let size = 1usize << 17;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    let mut vcpu = bytecode::Vcpu::new_root_with_powerbox(
        &prog,
        0,
        &[Value::I32(inst), Value::I32(mh)],
        Arc::clone(&back),
        &[],
        host,
    )
    .expect("root vcpu");
    // One `run()` suffices: the coroutine-module kernel has no thread.spawn / futex, and every §14
    // op is serviced internally, so the vCPU runs to `Done` without the host servicing any event.
    let r = match vcpu.run() {
        bytecode::VcpuEvent::Done(vals) => Ok(vals),
        bytecode::VcpuEvent::Trapped(t) => Err(t),
        _ => panic!("unexpected host event: coroutine-module needs none"),
    };
    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone.
    unsafe { std::alloc::dealloc(base, layout) };
    r
}

/// A powerbox-carrying resumable `Vcpu` builds + resumes 8 separate-module coroutines internally
/// (no host round-trip) — the folded sum matches the cooperative oracle.
#[test]
fn resumable_coroutine_module_matches_oracle() {
    let want = run_cooperative();
    assert_eq!(
        want,
        Ok(vec![Value::I64(600)]),
        "oracle: 8 × coroutine-module(75) = 600"
    );
    for i in 0..10 {
        assert_eq!(
            run_resumable(),
            want,
            "resumable spawn_coroutine_module != oracle (run {i})"
        );
    }
}
