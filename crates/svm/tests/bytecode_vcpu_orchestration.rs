//! THREADS.md step 4c-wasm — proof that the **host-orchestrated** resumable per-vCPU API
//! (`VcpuProgram` + `Vcpu` + `VcpuEvent`) correctly runs one guest's `thread.spawn`ed vCPUs when an
//! **external** host services every multi-vCPU event (spawn → start a vCPU, join → block on its
//! result, wait/notify → a futex). Here the host is a tiny `std::thread` orchestrator that plays
//! exactly the role the JS/Worker host will play in the browser — so this is the executable spec (and
//! differential oracle) for the wasm driver, with the cooperative engine as ground truth.
//!
//! The `unsafe` (borrowing host memory via `Region::shared`) lives in this embedder/test; the engine
//! stays `#![forbid(unsafe_code)]`.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use svm_interp::{bytecode, Region, Trap, Value};
use svm_text::parse_module;

// 8 vCPUs each `atomic.rmw.add` a shared counter 500× → 4000; the root then joins them. The host
// (below) creates each spawned vCPU on its own OS thread and blocks each join — so this is the same
// genuine parallelism as `drive_parallel`, but driven entirely through the public resumable API.
const THREADS: &str = r#"memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1) block3()
block2(v4: i64):
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br block1(v12)
block3():
  v13 = i64.const 0
  br block4(v13)
block4(v14: i64):
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 block5(v14) block6()
block5(v17: i64):
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br block4(v25)
block6():
  v26 = i64.const 0
  v27 = i64.atomic.load v26
  return v27
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 block2() block3(v1)
block3(v4: i64):
  v5 = i64.const 0
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br block1(v9)
block2():
  v10 = i64.const 0
  return v10
}
"#;

// Futex handoff (root parks via memory.wait until a spawned producer notifies it) → 987654.
const FUTEX: &str = r#"memory 16
func () -> (i64) {
block0():
  v3 = i64.const 0
  v4 = thread.spawn 1 v3 v3
  v0 = i64.const 0
  v1 = i32.const 0
  v2 = i64.const 1000000000
  v5 = i32.atomic.wait v0 v1 v2
  v6 = i64.const 8
  v7 = i64.atomic.load.acquire v6
  v8 = thread.join v4
  return v7
}
func (i64, i64) -> (i64) {
block0(vsp: i64, v0: i64):
  v1 = i64.const 8
  v2 = i64.const 987654
  i64.atomic.store.release v1 v2
  v3 = i64.const 0
  v4 = i32.const 1
  i32.atomic.store.release v3 v4
  v5 = i64.const 0
  v6 = i32.const 1
  v7 = atomic.notify v5 v6
  v8 = i64.const 0
  return v8
}
"#;

/// One parked futex waiter: a flag + a condvar (the host's stand-in for `memory.atomic.wait`).
struct Waiter {
    woken: Mutex<bool>,
    cv: Condvar,
}

/// The host's orchestration state, shared across the vCPU threads: the join rendezvous (`done`/`cv`)
/// and the futex (`futex`). Exactly the responsibilities the JS/Worker host carries in the browser.
struct Host {
    next_id: Mutex<u64>,
    done: Mutex<HashMap<u64, Result<Vec<Value>, Trap>>>,
    cv: Condvar,
    futex: Mutex<HashMap<u64, VecDeque<Arc<Waiter>>>>,
}

impl Host {
    fn new() -> Host {
        Host {
            next_id: Mutex::new(0),
            done: Mutex::new(HashMap::new()),
            cv: Condvar::new(),
            futex: Mutex::new(HashMap::new()),
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
    /// `memory.wait`: compare the futex word in the shared backing under the bucket lock, then park.
    fn wait(&self, back: &Region, addr: u64, expected: u64, width: u32, timeout: u64) -> i32 {
        let waiter = {
            let mut buckets = self.futex.lock().unwrap();
            if back.atomic_load(addr, width) != expected {
                return 1; // not-equal fast path
            }
            let w = Arc::new(Waiter {
                woken: Mutex::new(false),
                cv: Condvar::new(),
            });
            buckets.entry(addr).or_default().push_back(Arc::clone(&w));
            w
        };
        let (flag, res) = waiter
            .cv
            .wait_timeout_while(
                waiter.woken.lock().unwrap(),
                std::time::Duration::from_nanos(timeout),
                |w| !*w,
            )
            .unwrap();
        if *flag {
            0 // woken
        } else {
            debug_assert!(res.timed_out());
            let mut buckets = self.futex.lock().unwrap();
            if let Some(q) = buckets.get_mut(&addr) {
                q.retain(|x| !Arc::ptr_eq(x, &waiter));
            }
            2 // timed out
        }
    }
    /// `memory.notify`: wake up to `count` parked waiters on `addr`; return how many.
    fn notify(&self, addr: u64, count: i32) -> i32 {
        let want = count as u32;
        let mut buckets = self.futex.lock().unwrap();
        let mut woke = 0u32;
        if let Some(q) = buckets.get_mut(&addr) {
            while woke < want {
                let Some(w) = q.pop_front() else { break };
                *w.woken.lock().unwrap() = true;
                w.cv.notify_one();
                woke += 1;
            }
        }
        woke as i32
    }
}

/// Drive one `Vcpu` to completion, fanning each `thread.spawn` onto a fresh scoped OS thread and
/// blocking each `thread.join` / `memory.wait` on the host — the exact loop a JS/Worker host runs.
fn drive<'s, 'e>(
    scope: &'s std::thread::Scope<'s, 'e>,
    prog: &'e bytecode::VcpuProgram,
    back: Arc<Region>,
    host: &'e Host,
    mut vcpu: bytecode::Vcpu<'e>,
) -> Result<Vec<Value>, Trap> {
    let mut handles: Vec<u64> = Vec::new(); // local spawn handle (index) → global id
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => return Ok(vals),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::Spawn { func, sp, arg } => {
                let id = host.fresh_id();
                let child = bytecode::Vcpu::new_child(
                    prog,
                    func,
                    &[Value::I64(sp), Value::I64(arg)],
                    Arc::clone(&back),
                )
                .expect("child vcpu");
                let cback = Arc::clone(&back);
                scope.spawn(move || {
                    let r = drive(scope, prog, cback, host, child);
                    host.publish(id, r);
                });
                let handle = handles.len() as i32;
                handles.push(id);
                vcpu.deliver_handle(handle);
            }
            bytecode::VcpuEvent::Join { handle } => {
                let id = handles[handle as usize];
                vcpu.deliver_join(host.join(id));
            }
            bytecode::VcpuEvent::Wait {
                addr,
                expected,
                width,
                timeout,
            } => {
                let code = host.wait(&back, addr, expected, width, timeout);
                vcpu.deliver_code(code);
            }
            bytecode::VcpuEvent::Notify { addr, count } => {
                vcpu.deliver_code(host.notify(addr, count));
            }
            // §22 JIT events: out of scope for these compute/threads/futex kernels (covered by
            // `bytecode_vcpu_orchestration_jit.rs`) — never raised here.
            bytecode::VcpuEvent::JitInstall { .. }
            | bytecode::VcpuEvent::JitUninstall { .. }
            | bytecode::VcpuEvent::JitInvoke { .. } => {
                unreachable!("no JIT in the compute/threads/futex orchestration kernels")
            }
        }
    }
}

/// Run `src`'s function 0 via the resumable API under the `std::thread` host orchestrator.
fn run_orchestrated(src: &str) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).unwrap();
    let prog = bytecode::VcpuProgram::compile(&m).unwrap();

    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    let host = Host::new();
    let root = bytecode::Vcpu::new_root(&prog, 0, &[], Arc::clone(&back), &[]).unwrap();
    let r = std::thread::scope(|scope| drive(scope, &prog, Arc::clone(&back), &host, root));

    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    r
}

/// The cooperative oracle, for ground truth.
fn run_cooperative(src: &str) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).unwrap();
    let mut f = u64::MAX;
    bytecode::compile_and_run_capture(&m, 0, &[], &mut f, &[])
        .unwrap()
        .0
}

/// The resumable API, driven by an external `std::thread` host (the browser's JS/Worker host modelled
/// natively), runs the 8-vCPU counter kernel to the oracle's 4000 — stable across real-race repeats.
#[test]
fn orchestrated_threads_match_oracle() {
    let want = run_cooperative(THREADS);
    assert_eq!(want, Ok(vec![Value::I64(4000)]), "oracle");
    for i in 0..50 {
        assert_eq!(run_orchestrated(THREADS), want, "orchestrated run {i}");
    }
}

/// The host services `memory.wait`/`notify` (its futex): the root parks until a spawned producer
/// notifies it, then reads the handed-over payload → 987654, matching the oracle.
#[test]
fn orchestrated_futex_matches_oracle() {
    let want = run_cooperative(FUTEX);
    assert_eq!(want, Ok(vec![Value::I64(987654)]), "oracle");
    for i in 0..50 {
        assert_eq!(run_orchestrated(FUTEX), want, "orchestrated run {i}");
    }
}
