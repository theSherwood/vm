//! PROCESS.md §4 / S4 — a **cross-domain pipe**: a parent mints a pipe (`grant_pipe`), keeps the read
//! end, and re-grants the **write end** into a §14 child (`instantiate_granted`, op 8). The child
//! writes bytes to its granted end; after `join` the parent reads them from its read end. The FIFO is
//! `Arc`-shared, so the two domains see the same queue — the substrate half of `cmd1 | cmd2`.
//!
//! Both backends re-grant the pipe end through the **same** `Host::regrant_into_child` (the interp op-8
//! path and the JIT's `svm_run::grant_child_build` both route through it), so this is a cross-backend
//! differential.

use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

/// func 0 (parent, `(Instantiator, read_end, write_end)`): `instantiate_granted` a child (func 1) in a
/// 4 KiB carve, re-granting the **write end**; `join`; then read 2 bytes from the **read end** into
/// window offset 16 and encode `read_count * 65536 + byte0 * 256 + byte1`. The child writes `"hi"`
/// (`'h'=104`, `'i'=105`), so the parent reads count `2` and those bytes → `2*65536 + 104*256 + 105`
/// = `157801` — proving the bytes crossed the domain boundary through the shared FIFO.
///
/// func 1 (child, `(Instantiator, AddressSpace, Stream)`): write `"hi"` into its own window and
/// `Stream.write(0, 2)` through the **granted write end**, then return 7.
const SRC: &str = "memory 17\n\
func (i32, i32, i32) -> (i64) {\n\
block0(vinst: i32, vread: i32, vwrite: i32):\n\
  vwrite64 = i64.extend_i32_u vwrite\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 8 (i64, i64, i64, i64, i64) -> (i32) vinst (vwrite64, ventry, voff, vsl, vq)\n\
  vcr = cap.call 6 1 (i32) -> (i64) vinst (vch)\n\
  a16 = i64.const 16\n\
  vlen = i64.const 2\n\
  vrd = cap.call 0 0 (i64, i64) -> (i64) vread (a16, vlen)\n\
  vb0 = i32.load8_u a16\n\
  a17 = i64.const 17\n\
  vb1 = i32.load8_u a17\n\
  k256 = i32.const 256\n\
  k65536 = i32.const 65536\n\
  vrdi = i32.wrap_i64 vrd\n\
  t0 = i32.mul vrdi k65536\n\
  t1 = i32.mul vb0 k256\n\
  t2 = i32.add t0 t1\n\
  t3 = i32.add t2 vb1\n\
  vresult = i64.extend_i32_u t3\n\
  return vresult\n\
}\n\
func (i64, i64, i64) -> (i64) {\n\
block0(vci: i64, vca: i64, vcw: i64):\n\
  a0 = i64.const 0\n\
  ch = i32.const 104\n\
  i32.store8 a0 ch\n\
  a1 = i64.const 1\n\
  ci = i32.const 105\n\
  i32.store8 a1 ci\n\
  vwh = i32.wrap_i64 vcw\n\
  vlen = i64.const 2\n\
  vw = cap.call 0 1 (i64, i64) -> (i64) vwh (a0, vlen)\n\
  v7 = i64.const 7\n\
  return v7\n\
}\n";

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

fn run_interp() -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let (w, r) = host.grant_pipe();
    let mut fuel = 50_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(r), Value::I32(w)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    )
    .0
}

fn run_jit() -> JitOutcome {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let (w, r) = host.grant_pipe();
    compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[ih as i64, r as i64, w as i64],
        &[0u8; 128 << 10],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        Some(grant_hooks()),
    )
    .expect("jit")
    .0
}

#[test]
fn child_writes_pipe_parent_reads_matches_interp() {
    let ir = run_interp();
    let jo = run_jit();
    assert_eq!(
        ir,
        Ok(vec![Value::I64(157_801)]),
        "interp: parent read 'hi' the child wrote through the granted pipe end"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[157_801]),
        "jit: cross-domain pipe must match interp, got {jo:?}"
    );
}
