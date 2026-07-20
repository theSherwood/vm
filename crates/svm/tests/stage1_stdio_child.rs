//! Stage 1 (STAGE1.md) slice 2 — a **stdio-inherited child**: the shell spawns an applet that both
//! (a) reads a parent-seeded `argv` token and (b) writes it to a `stdout` the parent granted it, so
//! the child's output lands in the shell's own sink. This is a real external `echo`: argv in, bytes
//! out through inherited stdio, exit status back — the composition of slice 1's seed+wait with the
//! `instantiate_named` grant path (op 11).
//!
//! Shape is **BusyBox-multicall**: parent (func 0) and applet (func 1) are one module; the applet is
//! "another `argv[0]`". The parent lays a single grant record `{name_off,name_len,handle,flags}`
//! naming its `stdout` handle, seeds the token into the child's carve, `instantiate_named`s the applet
//! granting that stdout, `join`s, and returns the applet's status. The applet resolves `"stdout"` by
//! name (`cap.self.resolve`) and writes the seeded bytes to it. Differential interp==JIT; the output
//! varies with the seed, proving argv delivery, and both backends write into the same shared sink.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites (svm-jit's guard page is unix-only).
#![cfg(unix)]

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const WIN: usize = 128 << 10; // parent window: 128 KiB
const CARVE: u64 = 64 << 10; // applet carve at 64 KiB

/// Build the one-module program (parent func 0 + applet func 1) with `token` (3 bytes) seeded into
/// the applet's carve. The parent takes `(Instantiator, stdout_handle)`.
fn src(token: &[u8; 3]) -> String {
    // Seed the token into the applet's window (its offset 0 = parent window CARVE); the applet writes
    // its resolve-name at offset 200, so offsets 0..2 stay the argv.
    let seed: String = token
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let addr = CARVE + i as u64;
            format!("  q{i} = i64.const {addr}\n  c{i} = i32.const {b}\n  i32.store8 q{i} c{i}\n")
        })
        .collect();
    format!(
        r#"memory 17
func (i32, i32) -> (i64) {{
block0(vinst: i32, vout: i32):
  a0 = i64.const 0
  n100 = i32.const 100
  i32.store a0 n100
  a4 = i64.const 4
  n6 = i32.const 6
  i32.store a4 n6
  a8 = i64.const 8
  i32.store a8 vout
  a12 = i64.const 12
  z0 = i32.const 0
  i32.store a12 z0
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  p100 = i64.const 100
  i32.store8 p100 cs
  p101 = i64.const 101
  i32.store8 p101 ct
  p102 = i64.const 102
  i32.store8 p102 cd
  p103 = i64.const 103
  i32.store8 p103 co
  p104 = i64.const 104
  i32.store8 p104 cu
  p105 = i64.const 105
  i32.store8 p105 ct
{seed}  gp = i64.const 0
  gn = i64.const 1
  ent = i64.const 1
  off = i64.const {CARVE}
  sl = i64.const 16
  q = i64.const 0
  vch = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) vinst (gp, gn, ent, off, sl, q)
  r = cap.call 6 1 (i32) -> (i64) vinst (vch)
  return r
}}
func (i64) -> (i64) {{
block0(vci: i64):
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  a200 = i64.const 200
  i32.store8 a200 cs
  a201 = i64.const 201
  i32.store8 a201 ct
  a202 = i64.const 202
  i32.store8 a202 cd
  a203 = i64.const 203
  i32.store8 a203 co
  a204 = i64.const 204
  i32.store8 a204 cu
  a205 = i64.const 205
  i32.store8 a205 ct
  len6 = i64.const 6
  hout = cap.self.resolve a200 len6
  a0 = i64.const 0
  len3 = i64.const 3
  w = cap.call 0 1 (i64, i64) -> (i64) hout (a0, len3)
  return w
}}
"#
    )
}

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

/// (parent result, stdout bytes) on the interpreter for `token`.
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

/// (parent outcome, stdout bytes) on the JIT for `token`, with the named-grant hooks installed.
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

/// The applet echoes its seeded `argv` to the granted stdout and returns the byte count as status —
/// identically on both backends, with the output tracking the seed (so it's a real argv, not a
/// constant). This is `echo` as an external command: spawn with argv, inherit stdout, wait for `$?`.
#[test]
fn stdio_child_echoes_seeded_argv_to_granted_stdout() {
    for token in [b"Hi!", b"abc"] {
        let (ir, iout) = run_interp(token);
        let (jo, jout) = run_jit(token);
        assert_eq!(
            ir.expect("interp run ok"),
            vec![Value::I64(3)],
            "interp: applet status = bytes written for {token:?}"
        );
        assert_eq!(
            iout, token,
            "interp: applet wrote its seeded argv to inherited stdout"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[3]),
            "jit: applet status must be 3 for {token:?}, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: inherited-stdout bytes must match interp");
    }
}
