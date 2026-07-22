//! Stage 1 (STAGE1.md) — a **foreign-program external command**: unlike the same-module applets of
//! `stage1_stdio_child.rs`, here the command is a *separate* host-verified `Module` (a distinct
//! binary), spawned via `Instantiator.instantiate_module` (op 5). Separate-module children have no
//! stdio-grant op, so the shell uses the **parent-as-pager** model: the child writes its output into
//! its carve, and the parent — seeing the §14 nested-carve superset — reads those bytes and forwards
//! them to its own `stdout`. Argv in (seeded), output forwarded, exit status back.
//!
//! This is the general `exec` case (run an arbitrary program), complementing the BusyBox-applet path.
//! Differential interp==JIT; the forwarded bytes track the seed, and the forward length is the
//! child's own return value (`join`), so a real result drives real output.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites (svm-jit's guard page is unix-only).
#![cfg(unix)]

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const WIN: usize = 128 << 10;
const CARVE: u64 = 64 << 10;
const OUT_OFF: u64 = 8; // child copies argv here; parent forwards from CARVE + OUT_OFF

/// The foreign command module: reads a 3-byte argv the parent seeded at offset 0, copies it to
/// `OUT_OFF` (its "stdout buffer"), and returns the byte count. A standalone binary — its output
/// leaves via the shared carve, not a granted stream.
fn child_src() -> &'static str {
    "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  s0 = i64.const 0
  b0 = i32.load8_u s0
  d0 = i64.const 8
  i32.store8 d0 b0
  s1 = i64.const 1
  b1 = i32.load8_u s1
  d1 = i64.const 9
  i32.store8 d1 b1
  s2 = i64.const 2
  b2 = i32.load8_u s2
  d2 = i64.const 10
  i32.store8 d2 b2
  n = i64.const 3
  return n
  }
}
"
}

/// The parent "shell" (entry args `(Instantiator, Module, stdout)`): seed `token` into the child's
/// carve, `instantiate_module` the foreign binary, `join` for its byte count `n`, then forward `n`
/// bytes from `CARVE + OUT_OFF` to stdout and return `n`.
fn parent_src(token: &[u8; 3]) -> String {
    let seed: String = token
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let addr = CARVE + i as u64;
            format!("  q{i} = i64.const {addr}\n  c{i} = i32.const {b}\n  i32.store8 q{i} c{i}\n")
        })
        .collect();
    let fwd = CARVE + OUT_OFF;
    format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vout: i32) {{
{seed}  me = i64.extend_i32_s vmod
  ent = i64.const 0
  off = i64.const {CARVE}
  sl = i64.const 16
  qz = i64.const 0
  ch = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (me, ent, off, sl, qz)
  n = cap.call 6 1 (i32) -> (i64) vinst (ch)
  fp = i64.const {fwd}
  w = cap.call 0 1 (i64, i64) -> (i64) vout (fp, n)
  return w
  }}
}}
"
    )
}

fn run_interp(token: &[u8; 3]) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let parent = parse_module(&parent_src(token)).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let child = parse_module(child_src()).expect("parse child");
    verify_module(&child).expect("verify child");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let mh = host.grant_module(&child);
    let oh = host.grant_stream(StreamRole::Out);
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I32(mh), Value::I32(oh)],
        &mut fuel,
        &[0u8; WIN],
        0,
        &mut host,
    );
    (res, host.stdout_bytes())
}

fn run_jit(token: &[u8; 3]) -> (JitOutcome, Vec<u8>) {
    let parent = parse_module(&parent_src(token)).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let child = parse_module(child_src()).expect("parse child");
    verify_module(&child).expect("verify child");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let mh = host.grant_module(&child);
    let oh = host.grant_stream(StreamRole::Out);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &parent,
        0,
        &[ih as i64, mh as i64, oh as i64],
        &[0u8; WIN],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        Some(svm_run::module_resolver),
        None,
    )
    .expect("jit");
    (jo, host.stdout_bytes())
}

/// A foreign binary runs as an external command: it consumes its seeded argv and produces output the
/// parent forwards to stdout, returning the byte count as status — identically on both backends, with
/// the forwarded bytes tracking the seed.
#[test]
fn foreign_command_output_is_forwarded_to_stdout() {
    for token in [b"cat", b"XYZ"] {
        let (ir, iout) = run_interp(token);
        let (jo, jout) = run_jit(token);
        assert_eq!(
            ir.expect("interp run ok"),
            vec![Value::I64(3)],
            "interp: parent forwarded 3 bytes for {token:?}"
        );
        assert_eq!(
            iout, token,
            "interp: forwarded output = child's processed argv"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[3]),
            "jit: parent forwarded 3 bytes for {token:?}, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: forwarded output must match interp");
    }
}
