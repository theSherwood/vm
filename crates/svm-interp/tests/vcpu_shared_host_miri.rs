//! THREADS.md 4d — **Miri** verification of the shared powerbox on the **resumable `Vcpu`** path. The
//! differential test in `svm/tests/bytecode_vcpu_orchestration_caps.rs` proves it agrees with the
//! cooperative oracle; this proves the machinery — every vCPU thread dispatching `cap.call` through
//! one `Mutex<Host>` ([`Vcpu::with_shared_host`]) while touching the shared window — is **free of
//! data races / UB / provenance errors**. Miri's checker, not the iteration count, is the point, so
//! the kernel (2 workers) is small.
//!
//! Run: `cargo +nightly miri test -p svm-interp --test vcpu_shared_host_miri`

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use svm_interp::{bytecode, Host, Region, StreamRole, Trap, Value};
use svm_text::parse_module;

// 2 worker vCPUs each write "hi\n" to stdout via `cap.call` (handle threaded through block args) +
// bump a shared counter — real concurrent `cap.call` on the one `Mutex<Host>` across threads.
const CAPS: &str = r#"memory 16
data 0 "hi\n"
func (i32) -> (i64) {
block 0 (v0: i32) {
  vh0 = i64.extend_i32_u v0
  v1 = i64.const 0
  br 1(v1, vh0)
}
block 1 (vi: i64, vhh: i64) {
  v2 = i64.const 2
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
  v11 = i64.const 2
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
  vlen = i64.const 3
  vw = cap.call 0 1 (i64, i64) -> (i64) vhandle(vptr, vlen)
  v1 = i64.const 8
  v2 = i64.const 1
  v3 = i64.atomic.rmw.add v1 v2
  v4 = i64.const 0
  return v4
  }
}
"#;

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
            _ => panic!("unexpected event in the shared-host Miri kernel"),
        }
    }
}

#[test]
fn vcpu_shared_host_capcall_race_free_under_miri() {
    let m = parse_module(CAPS).unwrap();
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
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
    assert_eq!(r, Ok(vec![Value::I64(2)]));
    assert_eq!(host.stdout, b"hi\n".repeat(2));
}
