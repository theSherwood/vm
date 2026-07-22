//! THREADS.md 4d — **host I/O (`cap.call`) from every vCPU of a host-orchestrated resumable run**,
//! the way the JS/Worker host will drive it in the browser. Each `Vcpu` attaches the run's shared
//! powerbox ([`Vcpu::with_shared_host`], a `Mutex<Host>`), so a worker vCPU's `cap.call` dispatches
//! in-engine under the lock — the resumable counterpart of `drive_parallel`'s 4c-host model, with the
//! same property: each call locks only for its own dispatch, compute/atomics between calls stay
//! lock-free, and the host never services (or even sees) a capability event.
//!
//! The kernel is `bytecode_parallel_caps.rs`'s proven schedule-independent one: 8 worker vCPUs each
//! (a) write the **same** 5-byte line to stdout via `cap.call` and (b) `atomic.rmw.add` a shared
//! counter — so both the result (8) and the stdout bytes (`"tick\n"` × 8) are byte-identical to the
//! **cooperative** oracle no matter how the Workers interleave. The `unsafe` (borrowing host memory
//! via `Region::shared`) lives in this embedder/test.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use svm_interp::{bytecode, Host, Region, StreamRole, Trap, Value};
use svm_text::parse_module;

/// func 0 (root, param = stdout handle): spawn 8 workers (passing the handle), join them, return the
/// counter at mem[8]. func 1 (worker, args = sp, stdout handle): write "tick\n" then bump the counter.
const CAPS: &str = r#"memory 16
data 0 "tick\n"
func (i32) -> (i64) {
block 0 (v0: i32) {
  vh0 = i64.extend_i32_u v0
  v1 = i64.const 0
  br 1(v1, vh0)
}
block 1 (vi: i64, vhh: i64) {
  v2 = i64.const 8
  v3 = i64.lt_u vi v2
  br_if v3 2(vi, vhh) 3()
}
block 2 (vi2: i64, vhh2: i64) {
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vhh2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br 1(v9, vhh2)
}
block 3 () {
  v10 = i64.const 0
  br 4(v10)
}
block 4 (vj: i64) {
  v11 = i64.const 8
  v12 = i64.lt_u vj v11
  br_if v12 5(vj) 6()
}
block 5 (vj2: i64) {
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br 4(v20)
}
block 6 () {
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vh: i64) {
  vhandle = i32.wrap_i64 vh
  vptr = i64.const 0
  vlen = i64.const 5
  vw = cap.call 0 1 (i64, i64) -> (i64) vhandle(vptr, vlen)
  v1 = i64.const 8
  v2 = i64.const 1
  v3 = i64.atomic.rmw.add v1 v2
  v4 = i64.const 0
  return v4
  }
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

/// Drive one vCPU to completion — spawn → a scoped thread whose `Vcpu` **shares the powerbox**;
/// join → the rendezvous. No capability event exists to service: `cap.call` never reaches this loop.
fn drive<'s, 'e>(
    scope: &'s std::thread::Scope<'s, 'e>,
    prog: &'e bytecode::VcpuProgram,
    back: &'e Arc<Region>,
    pb: &'e Mutex<Host>,
    orch: &'e Orch,
    mut vcpu: bytecode::Vcpu<'e>,
) -> Result<Vec<Value>, Trap> {
    let mut handles: Vec<u64> = Vec::new();
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
                    Arc::clone(back),
                )
                .expect("child vcpu")
                .with_shared_host(pb);
                scope.spawn(move || {
                    let r = drive(scope, prog, back, pb, orch, child);
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
            _ => panic!("unexpected event in the shared-host caps kernel"),
        }
    }
}

/// Run the kernel through the resumable API with one shared powerbox; returns (result, stdout).
fn run_orchestrated() -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(CAPS).unwrap();
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    // Grant before sharing (deterministic handle), read `stdout` back after the run.
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::Out);
    let pb = Mutex::new(host);

    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this run's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    let orch = Orch::new();
    let root = bytecode::Vcpu::new_root(&prog, 0, &[Value::I32(h)], Arc::clone(&back), &[])
        .expect("root vcpu")
        .with_shared_host(&pb);
    let r = std::thread::scope(|scope| drive(scope, &prog, &back, &pb, &orch, root));

    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    let host = pb.into_inner().unwrap_or_else(|e| e.into_inner());
    (r, host.stdout)
}

/// Cooperative oracle: one shared host across all vCPUs (deterministic). Returns (result, stdout).
fn run_cooperative() -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(CAPS).unwrap();
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::Out);
    let mut f = u64::MAX;
    let r =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(h)], &mut f, &mut host).unwrap();
    (r, host.stdout)
}

/// 8 worker vCPUs do host I/O (`cap.call` stdout write) + atomics on one **shared** powerbox through
/// the resumable API — result and the (identical-line, schedule-independent) stdout match the oracle.
#[test]
fn orchestrated_shared_host_capcall_matches_oracle() {
    let (want_r, want_out) = run_cooperative();
    assert_eq!(
        want_r,
        Ok(vec![Value::I64(8)]),
        "oracle: 8 workers → counter 8"
    );
    assert_eq!(want_out, b"tick\n".repeat(8), "oracle stdout: 8 lines");

    // Real races on the shared host — a wrong lock/sharing would corrupt stdout or the counter.
    for i in 0..25 {
        let (got_r, got_out) = run_orchestrated();
        assert_eq!(got_r, want_r, "orchestrated result != oracle (run {i})");
        assert_eq!(got_out, want_out, "orchestrated stdout != oracle (run {i})");
    }
}
