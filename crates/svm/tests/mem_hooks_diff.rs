//! Three-backend parity gate for **memory-access hooks** (`Instance::with_mem_hooks` →
//! `svm_opt::instrument::instrument_mem_hooks`, HOOKS.md).
//!
//! The hooks design keeps every engine untouched — an instrumented module is an ordinary module —
//! so the §3 parity invariant must extend to the *event stream*: the tree-walker, the bytecode
//! engine, and the JIT run the same instrumented module and must report the **identical sequence**
//! of memory events (kind, address, width/span), produce the same guest-visible outcome as the
//! pristine module, and, on a faulting run, agree on a trace whose final event is the *attempted*
//! faulting access (hooks fire pre-access, pre-confinement-check).

use std::sync::{Arc, Mutex};
use svm_interp::Trap;
use svm_run::{instantiate, Backend, MemEvent, MemHookFn, RunConfig};
use svm_text::parse_module;

const BACKENDS: [Backend; 3] = [Backend::TreeWalk, Backend::Bytecode, Backend::Jit];

/// One of every event kind: scalar store/load with an immediate offset, a v128 load (the bytecode
/// engine routes v128 through its `Op::Eval` fallback — this pins that the hook still fires
/// there), bulk fill + copy, and all four atomics.
const SRC: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 64
  v1 = i64.const 7
  i64.store v0 v1 offset=8
  v2 = i32.const 170
  v3 = i64.const 32
  mem.fill v0 v2 v3
  v4 = i64.const 192
  mem.copy v4 v0 v3
  v5 = i32.const 1
  i32.atomic.store v0 v5
  v6 = i32.atomic.load v0
  v7 = i32.atomic.rmw.add v0 v5
  v8 = i32.atomic.cmpxchg v0 v5 v5
  v9 = v128.load v0
  v10 = i64.load v0 offset=8
  return v10
  }
}
"#;

/// The event stream `SRC` must produce, on every backend.
fn expected_events() -> Vec<MemEvent> {
    vec![
        MemEvent::Store { addr: 72, width: 8 },
        MemEvent::Fill { dst: 64, len: 32 },
        MemEvent::Copy {
            dst: 192,
            src: 64,
            len: 32,
        },
        MemEvent::AtomicStore { addr: 64, width: 4 },
        MemEvent::AtomicLoad { addr: 64, width: 4 },
        MemEvent::AtomicRmw { addr: 64, width: 4 },
        MemEvent::AtomicCmpxchg { addr: 64, width: 4 },
        MemEvent::Load {
            addr: 64,
            width: 16,
        },
        MemEvent::Load { addr: 72, width: 8 },
    ]
}

/// A recording hook: the shared event log plus the per-host handler factory
/// (`run_diff`-style hosts each build a fresh handler, all feeding one log).
fn recorder() -> (
    Arc<Mutex<Vec<MemEvent>>>,
    impl Fn() -> MemHookFn + Send + Sync + 'static,
) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let sink = events.clone();
    let make = move || -> MemHookFn {
        let sink = sink.clone();
        Box::new(move |ev| {
            sink.lock().unwrap().push(ev);
            Ok(())
        })
    };
    (events, make)
}

/// Run `src` hooked on `backend`, returning the recorded trace and the run result.
fn hooked_run(src: &str, backend: Backend) -> (Vec<MemEvent>, Result<svm_run::Run, String>) {
    let inst = instantiate(parse_module(src).expect("parse")).expect("instantiate");
    let (events, make) = recorder();
    let hooked = inst.with_mem_hooks(make).expect("with_mem_hooks");
    let run = hooked.run(backend, &RunConfig::default());
    let trace = events.lock().unwrap().clone();
    (trace, run)
}

#[test]
fn trace_is_identical_across_backends_and_the_result_is_unperturbed() {
    // The pristine module's outcome, per backend (the §3 invariant holds for it already).
    let pristine = instantiate(parse_module(SRC).expect("parse")).expect("instantiate");
    let base = pristine
        .run(Backend::TreeWalk, &RunConfig::default())
        .expect("pristine run");

    for backend in BACKENDS {
        let (trace, run) = hooked_run(SRC, backend);
        let run = run.unwrap_or_else(|e| panic!("hooked run failed on {backend:?}: {e}"));
        assert_eq!(
            run.outcome, base.outcome,
            "hooks must not perturb the guest-visible outcome ({backend:?})"
        );
        assert_eq!(
            trace,
            expected_events(),
            "event stream must be identical on every backend ({backend:?})"
        );
    }
}

/// An out-of-window store (widths cross the window end): every backend must fault, and the trace
/// must end with the *attempted* access — the hook fires before the confinement check.
const OOB: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 32
  v1 = i64.const 1
  i64.store v0 v1
  v2 = i64.const 65532
  i64.store v2 v1
  v3 = i64.const 0
  return v3
  }
}
"#;

#[test]
fn faulting_run_reports_the_attempted_access_last() {
    let expected = vec![
        MemEvent::Store { addr: 32, width: 8 },
        MemEvent::Store {
            addr: 65532,
            width: 8,
        },
    ];
    for backend in BACKENDS {
        let (trace, run) = hooked_run(OOB, backend);
        assert!(
            run.is_err(),
            "the OOB store must trap on {backend:?}, got {:?}",
            run.map(|r| r.outcome)
        );
        assert_eq!(
            trace, expected,
            "faulting trace must end at the attempted access ({backend:?})"
        );
    }
}

#[test]
fn a_hook_veto_aborts_the_run_identically_everywhere() {
    for backend in BACKENDS {
        let inst = instantiate(parse_module(SRC).expect("parse")).expect("instantiate");
        let (events, _) = recorder();
        let sink = events.clone();
        // Veto the third event (the copy): observe two, trap on the third.
        let hooked = inst
            .with_mem_hooks(move || -> MemHookFn {
                let sink = sink.clone();
                Box::new(move |ev| {
                    let mut seen = sink.lock().unwrap();
                    if seen.len() == 2 {
                        return Err(Trap::CapFault);
                    }
                    seen.push(ev);
                    Ok(())
                })
            })
            .expect("with_mem_hooks");
        let run = hooked.run(backend, &RunConfig::default());
        assert!(run.is_err(), "vetoed run must abort on {backend:?}");
        assert_eq!(
            events.lock().unwrap().clone(),
            expected_events()[..2].to_vec(),
            "the veto must land after exactly two observed events ({backend:?})"
        );
    }
}

/// A `thread.spawn` guest: the child vCPU stores a sentinel into shared memory, the parent joins and
/// atomic-loads it back. Hooked, this must (a) not crash and (b) observe **both** vCPUs' accesses —
/// the run's `Host` is shared across vCPUs (`Arc<Mutex<Host>>`), so the single hook handler is
/// invoked serialized under that lock (HOOKS.md §6). Threads are interpreter-only (the JIT reports
/// the ops unsupported), so this runs on the two interpreter backends. Cross-vCPU *order* is
/// schedule-dependent in general, but `join` orders the child's store before the parent's load, so
/// the multiset of events is fixed here.
const THREADED: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 43981
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  v3 = i64.const 0
  v4 = i64.atomic.load v3
  return v4
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  v1 = i64.const 0
  i64.atomic.store v1 v0
  return v0
  }
}
"#;

#[test]
fn multi_vcpu_guest_is_observed_without_crashing() {
    for backend in [Backend::TreeWalk, Backend::Bytecode] {
        let (trace, run) = hooked_run(THREADED, backend);
        let run = run.unwrap_or_else(|e| panic!("threaded hooked run failed on {backend:?}: {e}"));
        assert_eq!(
            run.outcome,
            svm_run::Outcome::Returned(vec![svm_run::Value::I64(43981)]),
            "the parent reads the child's shared-memory write ({backend:?})"
        );
        // The child's atomic store (from the spawned vCPU) and the parent's atomic load are both
        // observed by the one shared handler; `join` fixes their order here.
        assert_eq!(
            trace,
            vec![
                MemEvent::AtomicStore { addr: 0, width: 8 },
                MemEvent::AtomicLoad { addr: 0, width: 8 },
            ],
            "both vCPUs' accesses reach the hook ({backend:?})"
        );
    }
}
