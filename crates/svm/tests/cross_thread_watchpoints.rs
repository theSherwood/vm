//! Cross-thread **watchpoints** (DEBUGGING.md Milestone B — the "watchpoints across threads as a
//! headline test" that was the remaining gap). Watchpoints live in the run-shared `DebugShared`, and
//! every vCPU's per-op debug seam checks them, so a window-range watch fires in *whichever* thread
//! touches the range — reported with the faulting thread, the confined address, and read-vs-write.
//! These tests pin that across the matrix: write and read watches, thread attribution, **reading the
//! value each thread is about to write** at the watch, inspecting other threads while stopped at a
//! watch, and range precision.
//!
//! Driven under the deterministic scheduled debugger (`attach_scheduled`), so every run is reproducible
//! on one OS thread (no real-thread flakiness).

use svm_interp::{Inspector, Stop, StopReason, Value, WatchKind};
use svm_text::parse_module;

/// Two workers each do a non-atomic `load; add varg; store` on `mem[0]` (the classic lost-update
/// race); the root spawns both, joins, and reads the counter back. Worker `block0`: `vaddr=const 0`
/// [0], `vc=load` [1], `vn=add vc varg` [2], `store vaddr vn` [3], `vz=const 0` [4], `return vz`.
const RACY_COUNTER: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  va = i64.const 1
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 1 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 0
  vr = i64.load vaddr
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 0
  vc = i64.load vaddr
  vn = i64.add vc varg
  i64.store vaddr vn
  vz = i64.const 0
  return vz
  }
}
"#;

/// A **write**-watchpoint on the shared counter fires before *each* thread's store, reported against
/// the thread that is about to write — so watching a contended address surfaces every writer.
#[test]
fn write_watchpoint_fires_at_every_thread_write() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, vec![]);
    insp.set_watchpoint(0, 8, WatchKind::Write);

    let mut writers = Vec::new();
    loop {
        match insp.run_until_stop() {
            Stop::Break { reason, pc } => {
                assert!(
                    matches!(
                        reason,
                        StopReason::Watchpoint {
                            write: true,
                            addr: 0
                        }
                    ),
                    "a write-watch fires write=true at the watched addr, got {reason:?}"
                );
                assert_eq!(
                    (pc.func, pc.inst),
                    (1, 3),
                    "stopped before the worker's store"
                );
                writers.push(insp.stopped_task().expect("a thread is stopped"));
                insp.step(); // apply the store
            }
            Stop::Finished(r) => {
                assert_eq!(
                    r,
                    Ok(vec![Value::I64(2)]),
                    "both increments land (no race here)"
                );
                break;
            }
            Stop::Blocked => panic!("unexpected block"),
        }
    }
    writers.sort();
    assert_eq!(
        writers,
        vec![1, 2],
        "both worker threads' writes were caught, each once"
    );
}

/// A **read**-watchpoint fires on every thread's *load* of the watched range — the two workers' loads
/// and the root's final read-back.
#[test]
fn read_watchpoint_fires_at_every_thread_read() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, vec![]);
    insp.set_watchpoint(0, 8, WatchKind::Read);

    let mut readers = Vec::new();
    loop {
        match insp.run_until_stop() {
            Stop::Break { reason, .. } => {
                assert!(
                    matches!(
                        reason,
                        StopReason::Watchpoint {
                            write: false,
                            addr: 0
                        }
                    ),
                    "a read-watch fires write=false, got {reason:?}"
                );
                readers.push(insp.stopped_task().expect("a thread is stopped"));
                insp.step();
            }
            Stop::Finished(_) => break,
            Stop::Blocked => panic!("unexpected block"),
        }
    }
    readers.sort();
    assert_eq!(
        readers,
        vec![0, 1, 2],
        "both workers (1, 2) and the root (0) read the watched range"
    );
}

/// **The headline:** watch the contended counter and, at each thread's store, read the *value that
/// thread is about to write* — so you can stand on a shared variable and see exactly which thread
/// writes what, across the interleaving. Under the deterministic default order the increments are
/// sequential — worker 1 writes `1` (it read `0`), worker 2 writes `2` (it read `1`) — and the
/// watchpoint surfaces both, attributed to their threads, before they take effect.
///
/// (Freezing a *specific* racing interleaving — a watch under a fixed `find_schedule` witness plan —
/// is covered separately in `debug_witness_stepping.rs`; this one uses the live default schedule.)
#[test]
fn watchpoint_reads_each_threads_pending_write() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, vec![]);
    insp.set_watchpoint(0, 8, WatchKind::Write);

    let mut about_to_write = Vec::new();
    loop {
        match insp.run_until_stop() {
            Stop::Break { reason, .. } => {
                assert!(matches!(reason, StopReason::Watchpoint { write: true, .. }));
                let task = insp.stopped_task().expect("a thread is stopped");
                // `vn` (the add result being stored) is SSA value 4 in the worker block; read it on
                // the stopped thread *before* the store applies.
                let v = insp
                    .read_ir_value(0, 4)
                    .expect("the store's value operand is live");
                about_to_write.push((task, v));
                insp.step(); // apply the store, then continue to the next writer
            }
            Stop::Finished(r) => {
                assert_eq!(r, Ok(vec![Value::I64(2)]), "sequential increments → 2");
                break;
            }
            Stop::Blocked => panic!("unexpected block"),
        }
    }
    about_to_write.sort_by_key(|(t, _)| *t);
    assert_eq!(
        about_to_write,
        vec![(1, Value::I64(1)), (2, Value::I64(2))],
        "the watch caught each worker's store and the value it was about to write"
    );
}

/// While stopped at a watchpoint in one thread, `select_task` inspects any *other* live thread's stack
/// — cross-thread inspection composes with watchpoints, not just breakpoints.
#[test]
fn inspect_other_threads_at_a_watchpoint() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, vec![]);
    insp.set_watchpoint(0, 8, WatchKind::Write);

    assert!(matches!(
        insp.run_until_stop(),
        Stop::Break {
            reason: StopReason::Watchpoint { write: true, .. },
            ..
        }
    ));
    let stopped = insp.stopped_task().expect("stopped");
    let others: Vec<u64> = insp
        .threads()
        .into_iter()
        .filter(|&t| t != stopped)
        .collect();
    assert!(
        !others.is_empty(),
        "another worker is live while this one is at the watch"
    );
    for o in others {
        insp.select_task(o);
        let bt = insp.backtrace();
        assert!(
            bt.iter().all(|f| f.pc.func == 1),
            "the other thread is inside the worker function"
        );
    }
}

/// Range precision across threads: a watch on a *different* range (`[64, 72)`) is not tripped by the
/// workers' writes to `mem[0]` — the run finishes with no watch stop, exactly as the confined
/// `watch_hit` analysis dictates, regardless of which thread does the access.
#[test]
fn watchpoint_range_is_precise_across_threads() {
    let m = parse_module(RACY_COUNTER).expect("parse");
    let mut insp = Inspector::attach_scheduled(&m, 0, &[], 50_000_000, vec![]);
    insp.set_watchpoint(64, 8, WatchKind::ReadWrite); // a range no thread touches

    match insp.run_until_stop() {
        Stop::Finished(r) => assert_eq!(r, Ok(vec![Value::I64(2)]), "ran clean, no spurious watch"),
        other => panic!("a watch outside any access range must not fire: {other:?}"),
    }
}
