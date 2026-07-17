//! Stage 1 (STAGE1.md) — the **C-applet ABI**: an applet receives its `stdout` as an *entry argument*
//! (via `instantiate_granted`, op 8 — the handle is the child's 3rd arg) rather than resolving it by
//! name. This matters because the chibicc frontend's generic capability import takes the handle as the
//! **first C argument at runtime** (`codegen_ir.c` §7 generic import) and *cannot* emit
//! `cap.self.resolve` — so a compiled-C applet's natural shape is `applet(inst, addrspace, stdout_h)`
//! writing through `stdout_h`. This pins that path with a real payload: the applet echoes its
//! parent-**seeded** `argv` through the granted handle (existing op-8 tests hardcode the child's
//! output; here the output is data the parent delivered, proving argv-in + handle-as-arg together).
//!
//! Differential interp==JIT via `grant_child_build`; the output tracks the seed, so it's a real argv.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites (svm-jit's guard page is unix-only).
#![cfg(unix)]

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const WIN: usize = 128 << 10;
const CARVE: u64 = 64 << 10;

/// Parent (`(Instantiator, stdout)`) seeds `token` into the applet's carve, `instantiate_granted`s the
/// applet (func 1) re-granting its stdout, joins, and returns the applet's status. The applet
/// (`(Instantiator, AddressSpace, Stream)`) writes its seeded 3-byte argv through the granted Stream
/// handle it got as an argument, and returns the byte count.
fn src(token: &[u8; 3]) -> String {
    let seed: String = token
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let addr = CARVE + i as u64;
            format!("  q{i} = i64.const {addr}\n  c{i} = i32.const {b}\n  i32.store8 q{i} c{i}\n")
        })
        .collect();
    format!(
        "memory 17
func (i32, i32) -> (i64) {{
block0(vinst: i32, vout: i32):
{seed}  vgh = i64.extend_i32_u vout
  ventry = i64.const 1
  voff = i64.const {CARVE}
  vsl = i64.const 16
  vq = i64.const 0
  vch = cap.call 6 8 (i64, i64, i64, i64, i64) -> (i32) vinst (vgh, ventry, voff, vsl, vq)
  vres = cap.call 6 1 (i32) -> (i64) vinst (vch)
  return vres
}}
func (i64, i64, i64) -> (i64) {{
block0(vcinst: i64, vcas: i64, vcstream: i64):
  vsh = i32.wrap_i64 vcstream
  vptr = i64.const 0
  vlen = i64.const 3
  vw = cap.call 0 1 (i64, i64) -> (i64) vsh (vptr, vlen)
  return vw
}}
"
    )
}

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        release: svm_run::grant_child_release,
    }
}

fn run_interp(token: &[u8; 3]) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(&src(token)).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let oh = host.grant_stream(StreamRole::Out);
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(oh)],
        &mut fuel,
        &[0u8; WIN],
        0,
        &mut host,
    );
    (res, host.stdout_bytes())
}

fn run_jit(token: &[u8; 3]) -> (JitOutcome, Vec<u8>) {
    let m = parse_module(&src(token)).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let oh = host.grant_stream(StreamRole::Out);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[ih as i64, oh as i64],
        &[0u8; WIN],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        Some(grant_hooks()),
    )
    .expect("jit");
    (jo, host.stdout_bytes())
}

/// The applet writes its seeded argv through the stdout handle it received as an argument (not a
/// by-name resolve), returning the byte count — identically on both backends, output tracking the
/// seed. This is the ABI a chibicc-compiled applet emits: `applet(inst, addrspace, stdout_h)`.
#[test]
fn granted_applet_echoes_argv_through_handle_arg() {
    for token in [b"hey", b"yo!"] {
        let (ir, iout) = run_interp(token);
        let (jo, jout) = run_jit(token);
        assert_eq!(
            ir.expect("interp run ok"),
            vec![Value::I64(3)],
            "interp: applet status = bytes written for {token:?}"
        );
        assert_eq!(
            iout, token,
            "interp: applet echoed seeded argv via the handle arg"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[3]),
            "jit: applet status must be 3 for {token:?}, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: handle-arg echo must match interp");
    }
}
