//! THREADS.md 4c-domain §14-D2 — proof that the **host-orchestrated** resumable per-vCPU API runs §14
//! **confined executor children** (`Instantiator.instantiate` / `instantiate_module`) the way the
//! JS/Worker host will in the browser. The parent vCPU (carrying a powerbox, since §14 resolves its
//! `Instantiator` authority in-Vm) does all the authority-bearing work — carve validation, module
//! resolve + compile + push to the shared source, data-segment materialization — and surfaces a purely
//! mechanical [`VcpuEvent::Instantiate`]. The host's job is exactly the [`Spawn`] protocol: start a new
//! vCPU (here a scoped `std::thread`; in the browser a Worker) running
//! [`Vcpu::new_confined_child`] over `[win + carve, win + carve + 2^size_log2)`, and wire its
//! completion into `join`.
//!
//! Per DESIGN.md §14, *a sub-window is indistinguishable from a top-level window* — so the confined
//! child is literally a plain child whose window pointer is shifted and smaller (a fresh
//! `Region::shared` over the carve), with an attenuated powerbox and its own dispatch table both built
//! in-engine. Nesting composes with no special casing: a confined child's own `Instantiate` events are
//! relative to *its* window, and this host services them with the same arm (the depth-2 kernel).
//!
//! Differential vs the cooperative oracle; kernels are schedule-independent (every child computes the
//! same pure value, folded by `join`). The `unsafe` (aliasing views of one allocation via
//! `Region::shared` — the §13 shared-memory data plane) lives in this embedder/test.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use svm_interp::{bytecode, Host, Region, Trap, Value};
use svm_text::parse_module;

/// Root `(instantiator) -> sum`: instantiate 8 same-module children (func 1), each in its own 4 KiB
/// sub-window at `64 KiB + i*4 KiB`, then `join` each and sum. Each child returns 5 ⇒ 40.
const FANOUT: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  vi0 = i64.const 0
  br block1(vi0, v0)
block1(vi: i64, vinst: i32):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vinst) block3(vinst)
block2(vi2: i64, vinst2: i32):
  v4096 = i64.const 4096
  vofflo = i64.mul vi2 v4096
  v64k = i64.const 65536
  voff = i64.add v64k vofflo
  ventry = i64.const 1
  vslog = i64.const 12
  vquota = i64.const 0
  vh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst2 (ventry, voff, vslog, vquota)
  v4 = i64.const 4
  vholo = i64.mul vi2 v4
  v16 = i64.const 16
  vhoff = i64.add v16 vholo
  i32.store vhoff vh
  v1 = i64.const 1
  vinext = i64.add vi2 v1
  br block1(vinext, vinst2)
block3(vinst3: i32):
  vj0 = i64.const 0
  vs0 = i64.const 0
  br block4(vj0, vs0, vinst3)
block4(vj: i64, vs: i64, vinst4: i32):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj, vs, vinst4) block6(vs)
block5(vj2: i64, vs2: i64, vinst5: i32):
  v4b = i64.const 4
  vjlo = i64.mul vj2 v4b
  v16b = i64.const 16
  vjoff = i64.add v16b vjlo
  vhh = i32.load vjoff
  vr = cap.call 6 1 (i32) -> (i64) vinst5 (vhh)
  vsn = i64.add vs2 vr
  v1b = i64.const 1
  vjn = i64.add vj2 v1b
  br block4(vjn, vsn, vinst5)
block6(vs3: i64):
  return vs3
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 5
  return v1
}
"#;

/// Same fan-out, but each child (handed its own `Instantiator`) itself instantiates a grandchild
/// (func 2) in a 1 KiB sub-window of its own window — the host services the child's `Instantiate`
/// event with the same arm, its carve relative to the child's window. 8 × grandchild(9) = 72.
const NESTED: &str = r#"memory 17
func (i32) -> (i64) {
block0(v0: i32):
  vi0 = i64.const 0
  br block1(vi0, v0)
block1(vi: i64, vinst: i32):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vinst) block3(vinst)
block2(vi2: i64, vinst2: i32):
  v4096 = i64.const 4096
  vofflo = i64.mul vi2 v4096
  v64k = i64.const 65536
  voff = i64.add v64k vofflo
  ventry = i64.const 1
  vslog = i64.const 12
  vquota = i64.const 0
  vh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst2 (ventry, voff, vslog, vquota)
  v4 = i64.const 4
  vholo = i64.mul vi2 v4
  v16 = i64.const 16
  vhoff = i64.add v16 vholo
  i32.store vhoff vh
  v1 = i64.const 1
  vinext = i64.add vi2 v1
  br block1(vinext, vinst2)
block3(vinst3: i32):
  vj0 = i64.const 0
  vs0 = i64.const 0
  br block4(vj0, vs0, vinst3)
block4(vj: i64, vs: i64, vinst4: i32):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj, vs, vinst4) block6(vs)
block5(vj2: i64, vs2: i64, vinst5: i32):
  v4b = i64.const 4
  vjlo = i64.mul vj2 v4b
  v16b = i64.const 16
  vjoff = i64.add v16b vjlo
  vhh = i32.load vjoff
  vr = cap.call 6 1 (i32) -> (i64) vinst5 (vhh)
  vsn = i64.add vs2 vr
  v1b = i64.const 1
  vjn = i64.add vj2 v1b
  br block4(vjn, vsn, vinst5)
block6(vs3: i64):
  return vs3
}
func (i64) -> (i64) {
block0(v0: i64):
  vinst = i32.wrap_i64 v0
  ventry = i64.const 2
  voff = i64.const 0
  vslog = i64.const 10
  vquota = i64.const 0
  vgh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst (ventry, voff, vslog, vquota)
  vgr = cap.call 6 1 (i32) -> (i64) vinst (vgh)
  return vgr
}
func (i64) -> (i64) {
block0(v0: i64):
  v1 = i64.const 9
  return v1
}
"#;

/// The granted "plugin" module: 4 KiB window, a data segment `"K"` (75) at offset 0; its entry reads
/// that own data byte and returns it — proving compile + push + data materialization crossed threads.
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

/// Root `(instantiator, module) -> sum`: `instantiate_module` the granted module 8 times (one confined
/// 4 KiB child per slot at `64 KiB + i*4 KiB`), `join` each and sum. 8 × module-child(75) = 600.
const MODULE_FANOUT: &str = r#"memory 17
func (i32, i32) -> (i64) {
block0(vinst0: i32, vmod0: i32):
  vmod64 = i64.extend_i32_s vmod0
  vi0 = i64.const 0
  br block1(vi0, vinst0, vmod64)
block1(vi: i64, vinst: i32, vmod: i64):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vinst, vmod) block3(vinst)
block2(vi2: i64, vinst2: i32, vmod2: i64):
  v4096 = i64.const 4096
  vofflo = i64.mul vi2 v4096
  v64k = i64.const 65536
  voff = i64.add v64k vofflo
  ventry = i64.const 0
  vslog = i64.const 12
  vquota = i64.const 0
  vh = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst2 (vmod2, ventry, voff, vslog, vquota)
  v4 = i64.const 4
  vholo = i64.mul vi2 v4
  v16 = i64.const 16
  vhoff = i64.add v16 vholo
  i32.store vhoff vh
  v1 = i64.const 1
  vinext = i64.add vi2 v1
  br block1(vinext, vinst2, vmod2)
block3(vinst3: i32):
  vj0 = i64.const 0
  vs0 = i64.const 0
  br block4(vj0, vs0, vinst3)
block4(vj: i64, vs: i64, vinst4: i32):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj, vs, vinst4) block6(vs)
block5(vj2: i64, vs2: i64, vinst5: i32):
  v4b = i64.const 4
  vjlo = i64.mul vj2 v4b
  v16b = i64.const 16
  vjoff = i64.add v16b vjlo
  vhh = i32.load vjoff
  vr = cap.call 6 1 (i32) -> (i64) vinst5 (vhh)
  vsn = i64.add vs2 vr
  v1b = i64.const 1
  vjn = i64.add vj2 v1b
  br block4(vjn, vsn, vinst5)
block6(vs3: i64):
  return vs3
}
"#;

/// The join rendezvous the orchestrating host provides (the JS/Worker host's completion-slot analog).
struct Orch {
    next_id: Mutex<u64>,
    done: Mutex<HashMap<u64, Result<Vec<Value>, Trap>>>,
    cv: Condvar,
}

impl Orch {
    fn new() -> Orch {
        Orch {
            next_id: Mutex::new(0),
            done: Mutex::new(HashMap::new()),
            cv: Condvar::new(),
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
}

/// A window base pointer that crosses the scoped-thread hand-off (raw pointers aren't `Send`) while
/// keeping derived provenance (no int→ptr round-trip). Sound: only ever offset into the one live
/// allocation and handed to `Region::shared`.
#[derive(Clone, Copy)]
struct WinPtr(*mut u8);
// SAFETY: the pointee is the shared window (the §13 data plane); the pointer value is plain data.
unsafe impl Send for WinPtr {}

/// Drive one vCPU to completion, servicing each `Instantiate` event by starting a confined child on a
/// fresh scoped thread over `[win + carve, +2^size_log2)` — the exact loop the JS/Worker host runs.
/// `win` is *this* vCPU's window base pointer (nesting composes because a child's events are relative
/// to its own window).
fn drive<'s, 'e>(
    scope: &'s std::thread::Scope<'s, 'e>,
    prog: &'e bytecode::VcpuProgram,
    win: WinPtr,
    orch: &'e Orch,
    mut vcpu: bytecode::Vcpu<'e>,
) -> Result<Vec<Value>, Trap> {
    let mut handles: Vec<u64> = Vec::new(); // local handle (index) → global id
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => return Ok(vals),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::Instantiate {
                module,
                entry,
                carve,
                size_log2,
                fuel,
            } => {
                let id = orch.fresh_id();
                // SAFETY: the engine validated the carve inside this vCPU's window, which outlives the
                // scope; overlapping views of the one allocation are the §13 shared data plane (all
                // cross-thread access ordered by the spawn hand-off / the join rendezvous).
                let child_win = WinPtr(unsafe { win.0.add(carve as usize) });
                // SAFETY: as above — `2^size_log2` valid bytes at the validated carve.
                let back = Arc::new(unsafe { Region::shared(child_win.0, 1u64 << size_log2) });
                let child =
                    bytecode::Vcpu::new_confined_child(prog, module, entry, back, size_log2, fuel)
                        .expect("confined child vcpu");
                scope.spawn(move || {
                    let r = drive(scope, prog, child_win, orch, child);
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
            _ => panic!("unexpected event in the instantiate orchestration kernels"),
        }
    }
}

/// Run `src`'s function 0 through the resumable API under the `std::thread` host orchestrator, with
/// the powerbox + entry args `mk` builds (deterministic, so the oracle gets identical handles).
fn run_orchestrated(src: &str, mk: impl Fn() -> (Host, Vec<Value>)) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).unwrap();
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let (host, args) = mk();

    let size = 1usize << 17;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this run's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    let orch = Orch::new();
    let root =
        bytecode::Vcpu::new_root_with_powerbox(&prog, 0, &args, Arc::clone(&back), &[], host)
            .expect("root vcpu");
    let r = std::thread::scope(|scope| drive(scope, &prog, WinPtr(base), &orch, root));

    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    r
}

/// The cooperative oracle over the same powerbox + args.
fn run_cooperative(src: &str, mk: impl Fn() -> (Host, Vec<Value>)) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).unwrap();
    let (mut host, args) = mk();
    let mut f = 50_000_000u64;
    bytecode::compile_and_run_with_host(&m, 0, &args, &mut f, &mut host)
        .expect("bytecode engine drives §14 (cooperative)")
}

/// Powerbox: `Instantiator` over the whole 128 KiB window; args = `[inst]`.
fn inst_only() -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    (host, vec![Value::I32(inst)])
}

/// Powerbox: `Instantiator` + a `Module` grant for [`MODULE_CHILD`]; args = `[inst, mh]`.
fn inst_and_module() -> (Host, Vec<Value>) {
    let child = parse_module(MODULE_CHILD).expect("parse module child");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mh = host.grant_module(&child);
    (host, vec![Value::I32(inst), Value::I32(mh)])
}

/// 8 confined children, each a fresh `Vcpu` on its own thread over a carve region — joined and summed
/// to the cooperative oracle's value, across real-race repeats.
#[test]
fn orchestrated_instantiate_fanout_matches_oracle() {
    let want = run_cooperative(FANOUT, inst_only);
    assert_eq!(want, Ok(vec![Value::I64(40)]), "oracle: 8 × child(5) = 40");
    for i in 0..25 {
        assert_eq!(
            run_orchestrated(FANOUT, inst_only),
            want,
            "orchestrated instantiate != oracle (run {i})"
        );
    }
}

/// Depth-2: each confined child's own `Instantiate` event is serviced by the same host arm, its carve
/// relative to the child's window — confinement composes across threads/Workers with no special casing.
#[test]
fn orchestrated_instantiate_nested_matches_oracle() {
    let want = run_cooperative(NESTED, inst_only);
    assert_eq!(
        want,
        Ok(vec![Value::I64(72)]),
        "oracle: 8 × grandchild(9) = 72"
    );
    for i in 0..25 {
        assert_eq!(
            run_orchestrated(NESTED, inst_only),
            want,
            "orchestrated nested instantiate != oracle (run {i})"
        );
    }
}

/// 8 separate-module confined children: the parent resolved + compiled + pushed the granted module and
/// materialized its data segments before each event; each child runs the pushed module on its own
/// thread — folded to the oracle's value.
#[test]
fn orchestrated_instantiate_module_matches_oracle() {
    let want = run_cooperative(MODULE_FANOUT, inst_and_module);
    assert_eq!(
        want,
        Ok(vec![Value::I64(600)]),
        "oracle: 8 × module-child(75) = 600"
    );
    for i in 0..25 {
        assert_eq!(
            run_orchestrated(MODULE_FANOUT, inst_and_module),
            want,
            "orchestrated instantiate_module != oracle (run {i})"
        );
    }
}
