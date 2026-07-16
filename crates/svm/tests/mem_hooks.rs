//! The opt-in **memory-access hook** seam (`svm_interp::MemHooks`): embedder instrumentation
//! around guest loads/stores (memory-safety validators, tracers, profilers).
//!
//! What this gates:
//! - both interpreter engines (tree-walker + bytecode) fire the hooks for every guest data
//!   access — scalar loads/stores, atomics (RMW/cmpxchg report read+write), bulk ops — with
//!   identical event streams (the §18 parity discipline applied to the seam);
//! - the hook fires **after** confinement/protection checks and **before** the access, and a
//!   `false` return **vetoes**: the op raises `Trap::MemoryFault` with memory untouched;
//! - a host with no hooks installed behaves exactly as before (parity check);
//! - hooks fire through the production `run_with_host_fast` entry.

use std::sync::{Arc, Mutex};
use svm_interp::{bytecode, run_with_host, run_with_host_fast, Host, MemHooks, Trap, Value};
use svm_text::parse_module;

/// Records every hook event; optionally vetoes reads/writes at one address.
#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<(char, u64, u64)>>,
    veto_read_at: Option<u64>,
    veto_write_at: Option<u64>,
}

impl MemHooks for Recorder {
    fn read(&self, addr: u64, len: u64) -> bool {
        self.events.lock().unwrap().push(('r', addr, len));
        self.veto_read_at != Some(addr)
    }
    fn write(&self, addr: u64, len: u64) -> bool {
        self.events.lock().unwrap().push(('w', addr, len));
        self.veto_write_at != Some(addr)
    }
}

impl Recorder {
    fn events(&self) -> Vec<(char, u64, u64)> {
        self.events.lock().unwrap().clone()
    }
}

/// Store 7 at effective address 8+4, load it back. Hooks see the guest-relative effective
/// address (addr + offset).
const SCALAR: &str = r#"
memory 16
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 8
  v2 = i32.const 7
  i32.store v1 v2 offset=4
  v3 = i32.load v1 offset=4
  return v3
}
"#;

/// Fill [16,48) with 0xAA, copy it to [64,96), load an i32 back from 64.
const BULK: &str = r#"
memory 16
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 16
  v2 = i32.const 170
  v3 = i64.const 32
  mem.fill v1 v2 v3
  v4 = i64.const 64
  mem.copy v4 v1 v3
  v5 = i32.load v4
  return v5
}
"#;

/// atomic.store 5; rmw.add +5 (cell 10); cmpxchg(expected=5, repl=5) misses; atomic.load = 10.
const ATOMIC: &str = r#"
memory 16
func (i32) -> (i32) {
block0(v0: i32):
  v1 = i64.const 8
  v2 = i32.const 5
  i32.atomic.store v1 v2
  v3 = i32.atomic.rmw.add v1 v2
  v4 = i32.atomic.cmpxchg v1 v3 v2
  v5 = i32.atomic.load v1
  return v5
}
"#;

/// Run `src` on both engines with a fresh `rec`-configured host each; assert the results agree,
/// the bytecode engine actually drove the module (no fallback), and both engines produced the
/// **identical** hook event stream. Returns (result, events).
fn both_engines(
    src: &str,
    make_rec: impl Fn() -> Recorder,
) -> (Result<Vec<Value>, Trap>, Vec<(char, u64, u64)>) {
    let m = parse_module(src).expect("parse");

    let rec_tw = Arc::new(make_rec());
    let mut h_tw = Host::new();
    h_tw.set_mem_hooks(rec_tw.clone());
    let mut f_tw = 1_000_000u64;
    let tw = run_with_host(&m, 0, &[Value::I32(0)], &mut f_tw, &mut h_tw);

    let rec_bc = Arc::new(make_rec());
    let mut h_bc = Host::new();
    h_bc.set_mem_hooks(rec_bc.clone());
    let mut f_bc = 1_000_000u64;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(0)], &mut f_bc, &mut h_bc)
        .expect("bytecode engine must drive a pure compute+memory module");

    assert_eq!(tw, bc, "tree-walker != bytecode\n{src}");
    assert_eq!(
        rec_tw.events(),
        rec_bc.events(),
        "hook event streams diverge between engines\n{src}"
    );
    (tw, rec_tw.events())
}

#[test]
fn scalar_load_store_observed() {
    let (r, ev) = both_engines(SCALAR, Recorder::default);
    assert_eq!(r, Ok(vec![Value::I32(7)]));
    // Effective address 8 + offset 4 = 12, width 4; write then read.
    assert_eq!(ev, vec![('w', 12, 4), ('r', 12, 4)]);
}

#[test]
fn bulk_ops_observed_as_spans() {
    let (r, ev) = both_engines(BULK, Recorder::default);
    assert_eq!(r, Ok(vec![Value::I32(0xAAAA_AAAAu32 as i32)]));
    // fill = one write span; copy = one read span + one write span; then the scalar load.
    assert_eq!(
        ev,
        vec![('w', 16, 32), ('r', 16, 32), ('w', 64, 32), ('r', 64, 4)]
    );
}

#[test]
fn atomics_observed_rmw_reports_read_and_write() {
    let (r, ev) = both_engines(ATOMIC, Recorder::default);
    assert_eq!(r, Ok(vec![Value::I32(10)]));
    assert_eq!(
        ev,
        vec![
            ('w', 8, 4), // atomic.store
            ('r', 8, 4),
            ('w', 8, 4), // rmw.add: read + write
            ('r', 8, 4),
            ('w', 8, 4), // cmpxchg: may-write, reported up front (even on a miss)
            ('r', 8, 4)  // atomic.load
        ]
    );
}

#[test]
fn write_veto_faults_before_touching_memory() {
    let (r, ev) = both_engines(SCALAR, || Recorder {
        veto_write_at: Some(12),
        ..Recorder::default()
    });
    assert_eq!(r, Err(Trap::MemoryFault));
    // The vetoed write is the last event: the run died there, the load never happened.
    assert_eq!(ev, vec![('w', 12, 4)]);
}

#[test]
fn read_veto_faults_after_the_write_landed() {
    let (r, ev) = both_engines(SCALAR, || Recorder {
        veto_read_at: Some(12),
        ..Recorder::default()
    });
    assert_eq!(r, Err(Trap::MemoryFault));
    assert_eq!(ev, vec![('w', 12, 4), ('r', 12, 4)]);
}

/// A vetoed **write** must leave memory untouched: veto the copy's write span, then check (via a
/// second, hook-free run of a probe that loads the destination) that the bytes never moved —
/// here simply by asserting the veto killed the run before the load could return the copied value.
#[test]
fn veto_kills_bulk_copy() {
    let (r, ev) = both_engines(BULK, || Recorder {
        veto_write_at: Some(64),
        ..Recorder::default()
    });
    assert_eq!(r, Err(Trap::MemoryFault));
    assert_eq!(ev, vec![('w', 16, 32), ('r', 16, 32), ('w', 64, 32)]);
}

/// No-op hooks must not change results — and a hook-free host must behave exactly as before.
#[test]
fn noop_hooks_preserve_results() {
    struct Noop;
    impl MemHooks for Noop {}
    for src in [SCALAR, BULK, ATOMIC] {
        let m = parse_module(src).expect("parse");
        let mut f0 = 1_000_000u64;
        let plain = run_with_host(&m, 0, &[Value::I32(0)], &mut f0, &mut Host::new());
        let mut h = Host::new();
        h.set_mem_hooks(Arc::new(Noop));
        let mut f1 = 1_000_000u64;
        let hooked = run_with_host(&m, 0, &[Value::I32(0)], &mut f1, &mut h);
        assert_eq!(plain, hooked, "no-op hooks changed the result\n{src}");
        assert_eq!(f0, f1, "no-op hooks changed fuel accounting\n{src}");
    }
}

/// Root spawns a worker that stores 9 at address 32, joins it, and loads the value back — the
/// worker vCPU runs over a `fork_for_thread` view of the window, so this gates that hooks
/// propagate across `thread.spawn` (the join orders the two events deterministically).
const THREADED: &str = r#"
memory 16
func () -> (i32) {
block0():
  v0 = i64.const 4096
  v1 = i64.const 0
  v2 = thread.spawn 1 v0 v1
  v3 = thread.join v2
  v4 = i64.const 32
  v5 = i32.load v4
  return v5
}
func (i64, i64) -> (i64) {
block0(v0: i64, v1: i64):
  v2 = i64.const 32
  v3 = i32.const 9
  i32.store v2 v3
  v4 = i64.const 0
  return v4
}
"#;

#[test]
fn hooks_propagate_across_thread_spawn() {
    let m = parse_module(THREADED).expect("parse");
    let rec = Arc::new(Recorder::default());
    let mut h = Host::new();
    h.set_mem_hooks(rec.clone());
    let mut fuel = 1_000_000u64;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut h);
    assert_eq!(r, Ok(vec![Value::I32(9)]));
    // The worker's store (through its forked window view) then the root's post-join load.
    assert_eq!(rec.events(), vec![('w', 32, 4), ('r', 32, 4)]);
}

/// The production fast entry (`run_with_host_fast`, the bytecode engine) fires hooks too.
#[test]
fn hooks_fire_through_run_with_host_fast() {
    let m = parse_module(SCALAR).expect("parse");
    let rec = Arc::new(Recorder::default());
    let mut h = Host::new();
    h.set_mem_hooks(rec.clone());
    let mut fuel = 1_000_000u64;
    let r = run_with_host_fast(&m, 0, &[Value::I32(0)], &mut fuel, &mut h);
    assert_eq!(r, Ok(vec![Value::I32(7)]));
    assert_eq!(rec.events(), vec![('w', 12, 4), ('r', 12, 4)]);
}
