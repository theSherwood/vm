//! THREADS.md 4c-domain — **Miri** verification of §22 guest-JIT under the **parallel** driver. The
//! differential test in `svm/tests/bytecode_parallel_jit.rs` proves the parallel JIT path agrees with
//! the cooperative oracle; this proves the genuinely-new shared-`Domain` machinery — the
//! `Mutex<Vec<Arc<Compiled>>>` module source (concurrent `install`/`push`) and the `Box<[AtomicU64]>`
//! dispatch table (Release store on install, Acquire load on `call_indirect`) — is **free of data
//! races / UB / provenance errors** when real vCPU threads install + dispatch over one shared backing.
//! Miri's checker, not the iteration count, is the point, so the kernels (2 workers) and repeats are
//! small.
//!
//! The host JIT validator is stubbed inline (parse a fixed unit text) so this stays in the `svm-interp`
//! crate's Miri-buildable suite — `svm-run`'s real validator would drag in the Cranelift JIT Miri
//! can't compile.
//!
//! Run: `cargo +nightly miri test -p svm-interp --test parallel_jit_miri`

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, Value};
use svm_text::parse_module;

/// The unit every worker drives: `service() -> 7` — pure compute, no host/memory use.
const SERVICE: &str = r#"memory 16
func () -> (i32) {
block 0 () {
  v0 = i32.const 7
  return v0
  }
}
"#;

/// Stub blob validator (the role `svm-run`'s `jit_blob_validator` plays in the heavyweight suite):
/// ignore the opaque blob and hand back the fixed unit's funcs. Trusted test input, so no re-verify.
fn jit_validator(
    _bytes: &[u8],
    _mem_log2: Option<u8>,
    _symtab: &[u8],
) -> Result<Arc<[svm_ir::Func]>, i64> {
    match parse_module(SERVICE) {
        Ok(m) => Ok(m.funcs.into()),
        Err(_) => Err(-22),
    }
}

/// Root `(jit, code) -> counter`: pack both handles into the `thread.spawn` arg, spawn 2 workers, join,
/// return the shared counter at `mem[8]`. Worker: unpack, `Jit.invoke` the unit, atomically add its 7.
const INVOKE: &str = r#"memory 16
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vpacked = i64.or vchi vje
  vi0 = i64.const 0
  br 1(vi0, vpacked)
}
block 1 (vi: i64, vp: i64) {
  vn = i64.const 2
  vlt = i64.lt_u vi vn
  br_if vlt 2(vi, vp) 3()
}
block 2 (vi2: i64, vp2: i64) {
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vp2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br 1(v9, vp2)
}
block 3 () {
  vj0 = i64.const 0
  br 4(vj0)
}
block 4 (vj: i64) {
  vn2 = i64.const 2
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 5(vj) 6()
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
block 0 (vsp: i64, vp: i64) {
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
}
"#;

/// Same shape, but each worker `Jit.install`s the unit and `call_indirect`s its own raced slot —
/// concurrent installs into the shared dispatch table (the machinery Miri scrutinizes here).
const INSTALL: &str = r#"memory 16
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vpacked = i64.or vchi vje
  vi0 = i64.const 0
  br 1(vi0, vpacked)
}
block 1 (vi: i64, vp: i64) {
  vn = i64.const 2
  vlt = i64.lt_u vi vn
  br_if vlt 2(vi, vp) 3()
}
block 2 (vi2: i64, vp2: i64) {
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vp2
  v4 = i64.const 4
  v5 = i64.mul vi2 v4
  v6 = i64.const 16
  v7 = i64.add v6 v5
  i32.store v7 vt
  v8 = i64.const 1
  v9 = i64.add vi2 v8
  br 1(v9, vp2)
}
block 3 () {
  vj0 = i64.const 0
  br 4(vj0)
}
block 4 (vj: i64) {
  vn2 = i64.const 2
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 5(vj) 6()
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
block 0 (vsp: i64, vp: i64) {
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
}
"#;

/// A host granted the `Jit` cap (16-slot table) with the stub validator + one host-compiled unit.
fn jit_host() -> (Host, i32, i32) {
    let mut host = Host::new();
    host.set_jit_validator(jit_validator);
    let jit = host.grant_jit_with_table(Some(16), 4);
    let code = host
        .jit_compile(jit, b"service")
        .expect("no trap")
        .expect("compile ok")
        .handle;
    (host, jit, code)
}

fn run_parallel(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).unwrap();
    let (mut host, jit, code) = jit_host();
    let size = 1 << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
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

#[test]
fn parallel_jit_invoke_race_free_under_miri() {
    assert_eq!(run_parallel(INVOKE), Ok(vec![Value::I64(14)])); // 2 × invoke(7)
}

#[test]
fn parallel_jit_install_race_free_under_miri() {
    assert_eq!(run_parallel(INSTALL), Ok(vec![Value::I64(14)])); // 2 × install+call(7)
}
