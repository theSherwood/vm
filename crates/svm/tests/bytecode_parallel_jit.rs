//! THREADS.md 4c-domain — §22 guest-JIT (`Jit.install` / `Jit.invoke` + cross-module `call_indirect`)
//! under the **parallel** driver. The shared [`Domain`] (its module source + dispatch table) is
//! interior-mutable and thread-safe, so worker vCPUs running on real OS threads can install / invoke
//! against it concurrently while compute/atomics on the other vCPUs stay lock-free.
//!
//! Both kernels are written so the guest-observable result is **schedule-independent**: every worker
//! drives the *same* pure unit (`service() = 7`) and folds its return into a shared atomic counter, so
//! whatever order the installs/invokes interleave, the counter lands on `8 × 7 = 56` — byte-identical
//! to the **cooperative** single-threaded oracle (which already shares one `Domain` across its vCPUs,
//! deterministically). Internal slot/module indices differ per schedule, but the guest never observes
//! them. (A guest that *did* surface a raced install slot would be the documented opt-in
//! non-determinism of the parallel mode — not exercised here.)
//!
//! The `unsafe` (borrowing host memory via `Region::shared`) lives in this embedder/test, mirroring
//! `bytecode_parallel_caps.rs`.

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, Value};
use svm_run::grant_jit;
use svm_text::parse_module;
use svm_verify::verify_module;

/// The pre-compiled unit every worker drives: `service() -> 7`, a pure compute with no host/memory use,
/// so its result is identical on every vCPU regardless of when it runs.
const SERVICE: &str = r#"memory 16
func () -> (i32) {
block0():
  v0 = i32.const 7
  return v0
}
"#;

/// Root `(jit_handle, code_handle) -> counter`: pack both handles into the single `thread.spawn` arg
/// (`(code << 32) | jit`), spawn 8 workers (func 1), join them, return the shared counter at `mem[8]`.
/// Worker (func 1, args = sp, packed): unpack the handles, `Jit.invoke` the unit, atomically add its
/// return (7) to the counter. 8 workers ⇒ counter 56.
const INVOKE: &str = r#"memory 16
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vpacked = i64.or vchi vje
  vi0 = i64.const 0
  br block1(vi0, vpacked)
block1(vi: i64, vp: i64):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vp) block3()
block2(vi2: i64, vp2: i64):
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vp2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br block1(v9, vp2)
block3():
  vj0 = i64.const 0
  br block4(vj0)
block4(vj: i64):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj) block6()
block5(vj2: i64):
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br block4(v20)
block6():
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
}
func (i64, i64) -> (i64) {
block0(vsp: i64, vp: i64):
  vmask = i64.const 4294967295
  vjit64 = i64.and vp vmask
  vjit = i32.wrap_i64 vjit64
  vsh = i64.const 32
  vcode = i64.shr_u vp vsh
  vw = cap.call 11 1 (i64) -> (i32) vjit (vcode)
  vw64 = i64.extend_i32_u vw
  vc8 = i64.const 8
  vold = i64.atomic.rmw.add vc8 vw64
  vret = i64.const 0
  return vret
}
"#;

/// Same root/worker shape, but each worker `Jit.install`s the unit (→ a freshly raced table slot) and
/// `call_indirect`s **its own** slot — genuine concurrent installs into the shared dispatch table.
/// 8 distinct installs fit the 16-slot table (slot 0 = primary func); each returns 7 ⇒ counter 56.
const INSTALL: &str = r#"memory 16
func (i32, i32) -> (i64) {
block0(v0: i32, v1: i32):
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vpacked = i64.or vchi vje
  vi0 = i64.const 0
  br block1(vi0, vpacked)
block1(vi: i64, vp: i64):
  vn = i64.const 8
  vlt = i64.lt_u vi vn
  br_if vlt block2(vi, vp) block3()
block2(vi2: i64, vp2: i64):
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vp2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br block1(v9, vp2)
block3():
  vj0 = i64.const 0
  br block4(vj0)
block4(vj: i64):
  vn2 = i64.const 8
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 block5(vj) block6()
block5(vj2: i64):
  v13 = i64.const 4
  v14 = i64.mul vj2 v13
  v15 = i64.const 16
  v16 = i64.add v15 v14
  v17 = i32.load v16
  v18 = thread.join v17
  v19 = i64.const 1
  v20 = i64.add vj2 v19
  br block4(v20)
block6():
  v21 = i64.const 8
  v22 = i64.atomic.load v21
  return v22
}
func (i64, i64) -> (i64) {
block0(vsp: i64, vp: i64):
  vmask = i64.const 4294967295
  vjit64 = i64.and vp vmask
  vjit = i32.wrap_i64 vjit64
  vsh = i64.const 32
  vcode = i64.shr_u vp vsh
  vslot = cap.call 11 3 (i64) -> (i64) vjit (vcode)
  vslot32 = i32.wrap_i64 vslot
  vr = call_indirect () -> (i32) vslot32 ()
  vr64 = i64.extend_i32_u vr
  vc8 = i64.const 8
  vold = i64.atomic.rmw.add vc8 vr64
  vret = i64.const 0
  return vret
}
"#;

/// An 8-aligned zeroed buffer + a `Region::shared` over it; caller frees via the returned layout.
fn shared_window(size: usize) -> (Arc<Region>, *mut u8, std::alloc::Layout) {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here, used only as this window.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    (back, base, layout)
}

/// A fresh host granted the `Jit` cap (16-slot table) with `SERVICE` host-compiled into it; returns
/// `(host, jit_handle, code_handle)`. Granting/compiling into a fresh host is deterministic, so both
/// the parallel run and the oracle get identical handles.
fn host_with_unit(guest: &svm_ir::Module) -> (Host, i32, i32) {
    let mut host = Host::new();
    let jit = grant_jit(&mut host, guest, 4); // sets the blob validator; 2^4 = 16-slot table
    let svc = {
        let m = parse_module(SERVICE).expect("parse service");
        verify_module(&m).expect("verify service");
        svm_encode::encode_module(&m)
    };
    let code = host
        .jit_compile(jit, &svc)
        .expect("no trap")
        .expect("compile ok")
        .handle;
    (host, jit, code)
}

/// Cooperative oracle: one shared host across all vCPUs (deterministic).
fn run_cooperative(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).unwrap();
    verify_module(&m).expect("verify guest");
    let (mut host, jit, code) = host_with_unit(&m);
    let mut f = 50_000_000u64;
    bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(jit), Value::I32(code)],
        &mut f,
        &mut host,
    )
    .expect("bytecode engine drives §22 JIT (cooperative)")
}

/// Parallel: the same kernel over real OS threads sharing one `Domain` + powerbox.
fn run_parallel(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).unwrap();
    verify_module(&m).expect("verify guest");
    let (mut host, jit, code) = host_with_unit(&m);
    let (back, base, layout) = shared_window(1 << 16);
    let mut f = 50_000_000u64;
    let r = bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &[Value::I32(jit), Value::I32(code)],
        &mut f,
        &[],
        Arc::clone(&back),
        &mut host,
    )
    .expect("bytecode engine drives §22 JIT (parallel)")
    .0;
    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    r
}

/// 8 worker vCPUs concurrently `Jit.invoke` a pure unit on the **shared** domain — the folded counter
/// matches the cooperative oracle on every run (schedule-independent: each invoke returns 7).
#[test]
fn parallel_invoke_matches_oracle() {
    let want = run_cooperative(INVOKE);
    assert_eq!(want, Ok(vec![Value::I64(56)]), "oracle: 8 × invoke(7) = 56");
    for i in 0..50 {
        assert_eq!(
            run_parallel(INVOKE),
            want,
            "parallel invoke != oracle (run {i})"
        );
    }
}

/// 8 worker vCPUs concurrently `Jit.install` into the shared dispatch table and `call_indirect` their
/// own raced slot — real contention on the table; the folded counter still matches the oracle.
#[test]
fn parallel_install_call_indirect_matches_oracle() {
    let want = run_cooperative(INSTALL);
    assert_eq!(
        want,
        Ok(vec![Value::I64(56)]),
        "oracle: 8 × install+call(7) = 56"
    );
    for i in 0..50 {
        assert_eq!(
            run_parallel(INSTALL),
            want,
            "parallel install/call_indirect != oracle (run {i})"
        );
    }
}
