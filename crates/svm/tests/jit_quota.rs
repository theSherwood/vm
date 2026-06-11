//! §15 **spawn quota metering on the JIT** — the embedder-set fiber/vCPU caps enforced by the JIT's
//! `fiber_rt`/`os_thread_rt`, mirroring the interpreter (`svm/tests/quota.rs`). A guest exceeding its
//! quota traps cleanly (`FiberFault`/`ThreadFault`); the default quota = the anti-bomb ceilings, so an
//! unconfigured run is unchanged.
//!
//! Gated to the targets where the JIT fiber/thread runtime exists (`svm_fiber::supported()`).
//!
//! NB the JIT's fiber/vCPU tables don't include a *root* entry (the interpreter's do), so the same
//! quota value admits one more spawn here than on the interpreter — a pre-existing off-by-one; for DoS
//! *containment* what matters is the bound, not the exact threshold. Each test targets the JIT's own
//! counting.
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

/// Three `cont.new`s. The JIT fiber table holds no root, so `max_fibers = 2` admits the first two and
/// the **third** trips the quota (`FiberFault`); the default admits all three.
const FIBER_BOMB: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = ref.func 1
  v1 = i64.const 1024
  v2 = cont.new v0 v1
  v3 = cont.new v0 v1
  v4 = cont.new v0 v1
  v5 = i64.const 0
  return v5
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

/// Two `thread.spawn`s (no join). The JIT vCPU table holds no root, so `max_vcpus = 1` admits the
/// first spawn and the **second** trips the quota (`ThreadFault`); the default admits both.
const THREAD_BOMB: &str = r#"
memory 16
func () -> (i64) {
block0():
  v0 = i64.const 5
  v1 = thread.spawn 1 v0 v0
  v2 = thread.spawn 1 v0 v0
  return v0
}
func (i64, i64) -> (i64) {
block0(vsp: i64, varg: i64):
  return varg
}
"#;

#[test]
fn jit_vcpu_quota_traps() {
    match run(
        THREAD_BOMB,
        Quota {
            max_fibers: 1 << 16,
            max_vcpus: 1,
        },
    ) {
        JitOutcome::Trapped(TrapKind::ThreadFault) => {}
        other => panic!("expected ThreadFault at the vCPU quota, got {other:?}"),
    }
}

#[test]
fn jit_vcpu_quota_default_runs() {
    assert!(matches!(
        run(THREAD_BOMB, Quota::default()),
        JitOutcome::Returned(_)
    ));
}
