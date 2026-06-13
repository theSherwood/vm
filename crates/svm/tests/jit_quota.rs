//! §15 **spawn quota metering on the JIT** — the embedder-set fiber/vCPU caps enforced by the JIT's
//! `fiber_rt`/`os_thread_rt`, mirroring the interpreter (`svm/tests/quota.rs`). A guest exceeding its
//! quota traps cleanly (`FiberFault`/`ThreadFault`); the default quota = the anti-bomb ceilings, so an
//! unconfigured run is unchanged.
//!
//! The JIT now counts the **root** computation toward the quota (like the interpreter) and bounds
//! **concurrently-live** vCPUs (a finished thread frees its slot), so the same quota value means the
//! same thing on both backends — these use the same programs/expectations as the interpreter tests.
//!
//! Gated to the targets where the JIT fiber/thread runtime exists (`svm_fiber::supported()`).
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use core::ffi::c_void;
use svm_jit::{compile_and_run_with_host_fast, JitOutcome, Quota, TrapKind};
use svm_text::parse_module;
use svm_verify::verify_module;

/// A no-op cap thunk (these programs make no `cap.call`s; it's only needed by the `_fast` entry).
unsafe extern "C" fn noop_thunk(
    _ctx: *mut c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    _mem_reserved: u64,
    _type_id: u32,
    _op: u32,
    _handle: i32,
    _args: *const i64,
    _n_args: u64,
    _results: *mut i64,
    _n_results: u64,
    trap_out: *mut i64,
) {
    *trap_out = 0;
}
/// A resolver that claims nothing (so every `cap.call` would use the generic thunk).
unsafe extern "C" fn no_resolver(_t: u32, _o: u32, _na: u32, _nr: u32) -> *const c_void {
    core::ptr::null()
}

/// Run func 0 (no args) on the JIT under `quota` and return its outcome.
fn run(src: &str, quota: Quota) -> JitOutcome {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    compile_and_run_with_host_fast(
        &m,
        0,
        &[],
        noop_thunk,
        core::ptr::null_mut(),
        no_resolver,
        quota,
    )
    .expect("jit compile")
}

/// Two `cont.new`s. `max_fibers = 2` = the root + one fiber, so the **second** `cont.new` trips the
/// quota (`FiberFault`) — identical to the interpreter. The default admits both.
const FIBER_BOMB: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 1024
  v2 = cont.new v0 v1
  v3 = cont.new v0 v1
  v4 = i64.const 0
  return v4
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  return varg
}
"#;

#[test]
fn jit_fiber_quota_traps() {
    match run(
        FIBER_BOMB,
        Quota {
            max_fibers: 2,
            max_vcpus: 1 << 16,
        },
    ) {
        JitOutcome::Trapped(TrapKind::FiberFault) => {}
        other => panic!("expected FiberFault at the fiber quota, got {other:?}"),
    }
}

#[test]
fn jit_fiber_quota_default_runs() {
    assert!(matches!(
        run(FIBER_BOMB, Quota::default()),
        JitOutcome::Returned(_)
    ));
}

/// One `thread.spawn` + `join`. `max_vcpus = 1` = the root only, so the spawn (which would make a
/// second live vCPU) traps `ThreadFault` — identical to the interpreter. `max_vcpus = 2` admits it.
const THREAD_PROG: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 5
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  return varg
}
"#;

#[test]
fn jit_vcpu_quota_blocks_spawn() {
    match run(
        THREAD_PROG,
        Quota {
            max_fibers: 1 << 16,
            max_vcpus: 1,
        },
    ) {
        JitOutcome::Trapped(TrapKind::ThreadFault) => {}
        other => panic!("expected ThreadFault when the vCPU quota leaves no room, got {other:?}"),
    }
}

#[test]
fn jit_vcpu_quota_default_runs() {
    match run(THREAD_PROG, Quota::default()) {
        JitOutcome::Returned(v) => assert_eq!(v[0], 5),
        other => panic!("expected 5, got {other:?}"),
    }
    // A quota of 2 (root + one concurrent child) admits exactly this program.
    assert!(matches!(
        run(
            THREAD_PROG,
            Quota {
                max_fibers: 16,
                max_vcpus: 2,
            },
        ),
        JitOutcome::Returned(_)
    ));
}

/// **The fix for the cumulative-vs-concurrent issue:** spawn **and join** a vCPU 8 times in a loop.
/// Only one child is ever live, so `max_vcpus = 2` (root + one) must succeed — the previous
/// cumulative count (the never-shrinking handle table) would have false-trapped on the 3rd iteration.
const SPAWN_JOIN_LOOP: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 0
  br block1(v0)
block1(v1: i64):
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 block2(v1) block3()
block2(v4: i64):
  v5 = i64.const 7
  v6 = thread.spawn 1 v5 v5
  v7 = thread.join v6
  v8 = i64.const 1
  v9 = i64.add v4 v8
  br block1(v9)
block3():
  v10 = i64.const 42
  return v10
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  return varg
}
"#;

#[test]
fn jit_vcpu_quota_spawn_join_loop_is_concurrent() {
    // 8 spawn+join iterations, only one child live at a time ⇒ max_vcpus = 2 must not trap.
    match run(
        SPAWN_JOIN_LOOP,
        Quota {
            max_fibers: 16,
            max_vcpus: 2,
        },
    ) {
        JitOutcome::Returned(v) => assert_eq!(v[0], 42, "spawn-join loop completed"),
        other => panic!("spawn-join loop must not trip a concurrent quota, got {other:?}"),
    }
}

/// **The fiber quota is per-domain on the JIT too (D57 3b-ii)** — the shared fiber table carries
/// one `max_fibers` budget for root + every spawned vCPU, matching the interpreter's run-shared
/// registry (`fiber_migrate::fiber_quota_spans_vcpus`). With `max_fibers = 2` (root + one
/// creation), the root's `cont.new` fills the domain budget, so a *spawned vCPU's* `cont.new`
/// trips it — under the old per-vCPU tables the child's fresh table would have admitted it. One
/// more slot admits it, and the child's handle (1) continues the domain's numbering.
const FIBER_QUOTA_SPANS: &str = r#"
func () -> (i64) {
block0():
  v0 = ref.func 2
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = thread.spawn 1 v1 v1
  v4 = thread.join v3
  return v4
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  v0 = ref.func 2
  v1 = cont.new v0 varg
  v2 = i64.extend_i32_u v1
  return v2
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  return varg
}
"#;

#[test]
fn jit_fiber_quota_spans_vcpus() {
    match run(
        FIBER_QUOTA_SPANS,
        Quota {
            max_fibers: 2,
            max_vcpus: 1 << 16,
        },
    ) {
        JitOutcome::Trapped(TrapKind::FiberFault) => {}
        other => {
            panic!("the child's cont.new must trip the domain-wide fiber quota, got {other:?}")
        }
    }
    match run(
        FIBER_QUOTA_SPANS,
        Quota {
            max_fibers: 3,
            max_vcpus: 1 << 16,
        },
    ) {
        JitOutcome::Returned(v) => {
            assert_eq!(v[0], 1, "the child's handle continues the domain numbering")
        }
        other => panic!("one more slot must admit the child's cont.new, got {other:?}"),
    }
}
