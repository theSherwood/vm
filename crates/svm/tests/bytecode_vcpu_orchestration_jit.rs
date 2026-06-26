//! THREADS.md 4c-domain — proof that the **host-orchestrated** resumable per-vCPU API services §22
//! guest-JIT the way the JS/Worker host will in the browser. `Jit.install` / `Jit.uninstall` /
//! `Jit.invoke` surface as [`VcpuEvent`]s; the external host (which holds the powerbox) resolves the
//! unit's funcs and hands them back via `deliver_jit_*`, and the vCPU installs / invokes against the
//! **shared** [`Domain`] — so an install on one Worker's vCPU is visible to every other Worker's
//! `call_indirect` (the interior-mutable table from Increment A).
//!
//! Here the host is a tiny `std::thread` orchestrator playing exactly the JS/Worker role: it creates
//! each spawned vCPU on its own OS thread, blocks each join, and services each JIT event against one
//! shared powerbox. This is the executable spec (and differential oracle) for the wasm driver, with the
//! cooperative engine as ground truth. Both kernels fold to a schedule-independent counter (every
//! worker drives the pure unit `service() = 7`), so the result is byte-identical to the oracle.
//!
//! The `unsafe` (borrowing host memory via `Region::shared`) lives in this embedder/test; the engine
//! stays `#![forbid(unsafe_code)]`.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use svm_interp::{bytecode, Host, Region, Trap, Value};
use svm_run::grant_jit;
use svm_text::parse_module;
use svm_verify::verify_module;

/// The pre-compiled unit every worker drives: `service() -> 7` — pure compute, no host/memory use.
const SERVICE: &str = r#"memory 16
func () -> (i32) {
block0():
  v0 = i32.const 7
  return v0
}
"#;

/// Root `(jit, code) -> counter`: pack both handles into the single `thread.spawn` arg, spawn 8
/// workers (func 1), join them, return the shared counter at `mem[8]`. Worker: unpack, `Jit.invoke`
/// the unit, atomically add its return (7). 8 workers ⇒ counter 56.
const INVOKE: &str = r#"memory 16
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vpacked = i64.or vchi vje
  vi0 = i64.const 0
  br block1(vi0, vpacked)
block1(vi: i64, vp: i64):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vp) block3()
block2(vi2: i64, vp2: i64):
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vp2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br block1(v9, vp2)
block3():
  vj0 = i64.const 0
  br block4(vj0)
block4(vj: i64):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj) block6()
block5(vj2: i64):
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br block4(v20)
block6():
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
}
func (i64, i64) -> (i64) {
block0(vsp: i64, vp: i64):
  vmask = i64.const 4294967295
  vjit64 = i64.and vp vmask
  vjit = i32.wrap_i64 vjit64
  vsh = i64.const 32
  vcode = i64.shr_u vp vsh
  vw = cap.call 11 1 (i64) -> (i32) vjit (vcode)
  vw64 = i64.extend_i32_u vw
  vc8 = i64.const 8
  vold = i64.atomic.rmw.add vc8 vw64
  vret = i64.const 0
  return vret
}
"#;

/// Same shape, but each worker `Jit.install`s the unit (→ a freshly raced table slot) and
/// `call_indirect`s **its own** slot — concurrent installs into the shared dispatch table across the
/// orchestrated vCPUs; each returns 7 ⇒ counter 56.
const INSTALL: &str = r#"memory 16
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vpacked = i64.or vchi vje
  vi0 = i64.const 0
  br block1(vi0, vpacked)
block1(vi: i64, vp: i64):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vp) block3()
block2(vi2: i64, vp2: i64):
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vp2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br block1(v9, vp2)
block3():
  vj0 = i64.const 0
  br block4(vj0)
block4(vj: i64):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj) block6()
block5(vj2: i64):
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br block4(v20)
block6():
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
}
func (i64, i64) -> (i64) {
block0(vsp: i64, vp: i64):
  vmask = i64.const 4294967295
  vjit64 = i64.and vp vmask
  vjit = i32.wrap_i64 vjit64
  vsh = i64.const 32
  vcode = i64.shr_u vp vsh
  vslot = cap.call 11 3 (i64) -> (i64) vjit (vcode)
  vslot32 = i32.wrap_i64 vslot
  vr = call_indirect () -> (i32) vslot32 ()
  vr64 = i64.extend_i32_u vr
  vc8 = i64.const 8
  vold = i64.atomic.rmw.add vc8 vr64
  vret = i64.const 0
  return vret
}
"#;

/// The orchestrator's coordination state, shared across the vCPU threads: the join rendezvous
/// (`done`/`cv`) and the SVM **powerbox** (`pb`) it services JIT events against. Exactly the
/// responsibilities the JS/Worker host carries in the browser.
struct Orch {
    next_id: Mutex<u64>,
    done: Mutex<HashMap<u64, Result<Vec<Value>, Trap>>>,
    cv: Condvar,
    pb: Mutex<Host>,
}

impl Orch {
    fn new(pb: Host) -> Orch {
        Orch {
            next_id: Mutex::new(0),
            done: Mutex::new(HashMap::new()),
            cv: Condvar::new(),
            pb: Mutex::new(pb),
        }
    }
    fn fresh_id(&self) -> u64 {
        let mut n = self.next_id.lock().unwrap();
        let id = *n;
        *n += 1;
        id
    }
    fn publish(&self, id: u64, r: Result<Vec<Value>, Trap>) {
        self.done.lock().unwrap().insert(id, r);
        self.cv.notify_all();
    }
    fn join(&self, id: u64) -> Result<Vec<Value>, Trap> {
        let mut g = self.done.lock().unwrap();
        loop {
            if let Some(r) = g.remove(&id) {
                return r;
            }
            g = self.cv.wait(g).unwrap();
        }
    }
    /// Resolve a code-handle's unit funcs under authority `handle` (the install/invoke service):
    /// a forged / cross-domain / wrong-type handle is an inert `CapFault` → trap.
    fn resolve_unit(&self, handle: i32, code: i32) -> Result<Arc<[svm_ir::Func]>, Trap> {
        let g = self.pb.lock().unwrap();
        let domain = g.resolve_jit_domain(handle)?;
        let (cd, cu) = g.resolve_jit_code(code)?;
        if cd != domain {
            return Err(Trap::CapFault);
        }
        g.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
    }
    /// Authority check for `uninstall`: a forged/wrong-type domain handle traps.
    fn check_authority(&self, handle: i32) -> Result<(), Trap> {
        self.pb
            .lock()
            .unwrap()
            .resolve_jit_domain(handle)
            .map(|_| ())
    }
}

/// Drive one `Vcpu` to completion, fanning each `thread.spawn` onto a fresh scoped OS thread, blocking
/// each `thread.join`, and servicing each §22 JIT event against the shared powerbox — the exact loop a
/// JS/Worker host runs.
fn drive<'s, 'e>(
    scope: &'s std::thread::Scope<'s, 'e>,
    prog: &'e bytecode::VcpuProgram,
    back: Arc<Region>,
    orch: &'e Orch,
    mut vcpu: bytecode::Vcpu<'e>,
) -> Result<Vec<Value>, Trap> {
    let mut handles: Vec<u64> = Vec::new(); // local spawn handle (index) → global id
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => return Ok(vals),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::Spawn { func, sp, arg } => {
                let id = orch.fresh_id();
                let child = bytecode::Vcpu::new_child(
                    prog,
                    func,
                    &[Value::I64(sp), Value::I64(arg)],
                    Arc::clone(&back),
                )
                .expect("child vcpu");
                let cback = Arc::clone(&back);
                scope.spawn(move || {
                    let r = drive(scope, prog, cback, orch, child);
                    orch.publish(id, r);
                });
                let handle = handles.len() as i32;
                handles.push(id);
                vcpu.deliver_handle(handle);
            }
            bytecode::VcpuEvent::Join { handle } => {
                let id = handles[handle as usize];
                vcpu.deliver_join(orch.join(id));
            }
            bytecode::VcpuEvent::JitInstall { handle, code } => {
                vcpu.deliver_jit_install(orch.resolve_unit(handle, code));
            }
            bytecode::VcpuEvent::JitUninstall { handle, slot: _ } => {
                vcpu.deliver_jit_uninstall(orch.check_authority(handle));
            }
            bytecode::VcpuEvent::JitInvoke {
                handle,
                code,
                argv: _,
                params: _,
                results: _,
            } => {
                vcpu.deliver_jit_invoke(orch.resolve_unit(handle, code));
            }
            // These kernels use only spawn/join + JIT; wait/notify never arise here.
            bytecode::VcpuEvent::Wait { .. } | bytecode::VcpuEvent::Notify { .. } => {
                panic!("unexpected wait/notify event in JIT orchestration kernel")
            }
        }
    }
}

/// A fresh powerbox granted the `Jit` cap (16-slot table) with `SERVICE` host-compiled into it;
/// returns `(host, jit_handle, code_handle)`. Deterministic, so the run and the oracle get identical
/// handles.
fn powerbox_with_unit(guest: &svm_ir::Module) -> (Host, i32, i32) {
    let mut host = Host::new();
    let jit = grant_jit(&mut host, guest, 4); // sets the blob validator; 2^4 = 16-slot table
    let svc = {
        let m = parse_module(SERVICE).expect("parse service");
        verify_module(&m).expect("verify service");
        svm_encode::encode_module(&m)
    };
    let code = host
        .jit_compile(jit, &svc)
        .expect("no trap")
        .expect("compile ok")
        .handle;
    (host, jit, code)
}

/// Run `src`'s function 0 via the resumable API under the `std::thread` host orchestrator.
fn run_orchestrated(src: &str) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).unwrap();
    verify_module(&m).expect("verify guest");
    let (pb, jit, code) = powerbox_with_unit(&m);
    // Reserve the same `call_indirect` table the powerbox granted, so guest install lands in-range.
    let prog = bytecode::VcpuProgram::compile_with_jit_table(&m, pb.jit_table_log2()).unwrap();

    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    let orch = Orch::new(pb);
    let root = bytecode::Vcpu::new_root(
        &prog,
        0,
        &[Value::I32(jit), Value::I32(code)],
        Arc::clone(&back),
        &[],
    )
    .unwrap();
    let r = std::thread::scope(|scope| drive(scope, &prog, Arc::clone(&back), &orch, root));

    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    r
}

/// The cooperative oracle, for ground truth (one shared powerbox across all vCPUs, deterministic).
fn run_cooperative(src: &str) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).unwrap();
    verify_module(&m).expect("verify guest");
    let (mut host, jit, code) = powerbox_with_unit(&m);
    let mut f = 50_000_000u64;
    bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(jit), Value::I32(code)],
        &mut f,
        &mut host,
    )
    .expect("bytecode engine drives §22 JIT (cooperative)")
}

/// The resumable API, driven by an external `std::thread` host (the browser's JS/Worker host modelled
/// natively): 8 vCPUs each `Jit.invoke` the pure unit on the **shared** domain → the oracle's 56,
/// stable across real-race repeats.
#[test]
fn orchestrated_jit_invoke_matches_oracle() {
    let want = run_cooperative(INVOKE);
    assert_eq!(want, Ok(vec![Value::I64(56)]), "oracle: 8 × invoke(7) = 56");
    for i in 0..50 {
        assert_eq!(
            run_orchestrated(INVOKE),
            want,
            "orchestrated invoke run {i}"
        );
    }
}

/// 8 orchestrated vCPUs each `Jit.install` into the shared dispatch table and `call_indirect` their own
/// raced slot — install on one Worker's vCPU is visible to that vCPU's later dispatch through the
/// shared domain; the folded counter still matches the oracle.
#[test]
fn orchestrated_jit_install_matches_oracle() {
    let want = run_cooperative(INSTALL);
    assert_eq!(
        want,
        Ok(vec![Value::I64(56)]),
        "oracle: 8 × install+call(7) = 56"
    );
    for i in 0..50 {
        assert_eq!(
            run_orchestrated(INSTALL),
            want,
            "orchestrated install run {i}"
        );
    }
}
