//! §15 **spawn quota metering** — the embedder caps how many fibers / concurrently-live vCPUs a guest
//! may create, *below* the fixed anti-bomb ceilings, and a guest that exceeds it traps cleanly
//! (`FiberFault` / `ThreadFault`). This is DoS *containment* policy (the §5 fuel kill-path bounds
//! runaway *execution*; the quota bounds runaway *spawning*). Default quota = the hard ceilings, so an
//! unconfigured run is unchanged. These pin the interpreter (the reference executor); the JIT enforces
//! the same quota (see `jit_diff`'s quota tests / `svm-run`).

use svm_interp::{run_with_host, Host, Quota, Trap, Value};
use svm_text::parse_module;

fn module(src: &str) -> svm_ir::Module {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    m
}

/// Run func 0 (no args) under `quota` (None = default) and return the single i64 result or the trap.
fn run_q(src: &str, quota: Option<Quota>) -> Result<i64, Trap> {
    let m = module(src);
    let mut host = Host::new();
    if let Some(q) = quota {
        host.set_quota(q);
    }
    let mut fuel = 10_000_000u64;
    match run_with_host(&m, 0, &[], &mut fuel, &mut host) {
        Ok(vals) => match vals.as_slice() {
            [Value::I64(v)] => Ok(*v),
            other => panic!("expected one i64, got {other:?}"),
        },
        Err(t) => Err(t),
    }
}

/// Two `cont.new`s create two fibers (atop the root). `max_fibers = 2` lets the root + one fiber
/// exist, so the **second** `cont.new` trips the quota (`FiberFault`); the default quota runs it.
const FIBER_BOMB: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 1024
  v2 = cont.new v0 v1
  v3 = cont.new v0 v1
  v4 = i64.const 0
  return v4
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
"#;

#[test]
fn fiber_quota_traps_at_limit() {
    // max_fibers = 2: root fiber + the first cont.new = 2; the second cont.new exceeds it.
    assert_eq!(
        run_q(
            FIBER_BOMB,
            Some(Quota {
                max_fibers: 2,
                max_vcpus: 1 << 16,
            })
        ),
        Err(Trap::FiberFault),
        "second cont.new must trap at the fiber quota"
    );
}

#[test]
fn fiber_quota_default_runs() {
    // The default quota (the hard ceiling) admits both fibers.
    assert_eq!(run_q(FIBER_BOMB, None), Ok(0));
    // A generous explicit quota also runs.
    assert_eq!(
        run_q(
            FIBER_BOMB,
            Some(Quota {
                max_fibers: 10,
                max_vcpus: 10,
            })
        ),
        Ok(0)
    );
}

/// A program that spawns one vCPU, joins it, and returns its result (5).
const THREAD_PROG: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 5
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
"#;

#[test]
fn vcpu_quota_blocks_spawn() {
    // max_vcpus = 1: only the root vCPU is allowed, so the first `thread.spawn` (which would make a
    // second live vCPU) traps `ThreadFault` *before* a child is created — deterministic, no survivors.
    assert_eq!(
        run_q(
            THREAD_PROG,
            Some(Quota {
                max_fibers: 1 << 16,
                max_vcpus: 1,
            })
        ),
        Err(Trap::ThreadFault),
        "thread.spawn must trap when the vCPU quota leaves no room"
    );
}

#[test]
fn vcpu_quota_default_runs() {
    // Default quota: the spawn + join completes and returns the child's result.
    assert_eq!(run_q(THREAD_PROG, None), Ok(5));
    // A quota of 2 (root + one child) admits exactly this program.
    assert_eq!(
        run_q(
            THREAD_PROG,
            Some(Quota {
                max_fibers: 16,
                max_vcpus: 2,
            })
        ),
        Ok(5)
    );
}

/// The vCPU quota bounds **concurrent** liveness, not cumulative spawns: spawn+join 8 times (only one
/// child ever live), which `max_vcpus = 2` (root + one) must admit. (Pins parity with the JIT, whose
/// concurrent-live counter was fixed to match this.)
#[test]
fn vcpu_quota_spawn_join_loop_is_concurrent() {
    let src = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 2(v1) 3()
}
block 2 (v4: i64) {
  v5 = i64.const 7
  v6 = thread.spawn 1 v5 v5
  v7 = thread.join v6
  v8 = i64.const 1
  v9 = i64.add v4 v8
  br 1(v9)
}
block 3 () {
  v10 = i64.const 42
  return v10
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
"#;
    assert_eq!(
        run_q(
            src,
            Some(Quota {
                max_fibers: 16,
                max_vcpus: 2,
            })
        ),
        Ok(42)
    );
}

/// A quota only *tightens* — `set_quota` clamps each limit to the hard ceiling, so a guest can't raise
/// it past the anti-bomb bound by asking for more.
#[test]
fn quota_clamps_to_ceiling() {
    let mut host = Host::new();
    host.set_quota(Quota {
        max_fibers: usize::MAX,
        max_vcpus: usize::MAX,
    });
    let q = host.quota();
    // A `usize::MAX` request clamps to exactly the hard ceiling. Assert against the real constants so
    // this can't silently drift when a ceiling changes (max_fibers is 1<<24 since the handle widening).
    assert_eq!(
        q.max_fibers,
        svm_ir::MAX_FIBERS,
        "max_fibers clamped to ceiling"
    );
    assert_eq!(
        q.max_vcpus,
        svm_ir::MAX_VCPUS,
        "max_vcpus clamped to ceiling"
    );
    // And never below 1 (the root must exist).
    host.set_quota(Quota {
        max_fibers: 0,
        max_vcpus: 0,
    });
    let q = host.quota();
    assert_eq!(q.max_fibers, 1);
    assert_eq!(q.max_vcpus, 1);
}
