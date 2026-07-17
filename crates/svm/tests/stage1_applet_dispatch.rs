//! Stage 1 (STAGE1.md) slice 3 — **exit-status fidelity across a multi-applet binary**: one module
//! carries several "external commands" as applet entries (`true` → 0, `false` → 1, `echo` → writes
//! its seeded argv and returns the byte count), and a parent "shell" spawns a chosen applet, inherits
//! stdout into it, `join`s, and returns its status. Spawning different applets yields different
//! `(stdout, status)` pairs — the guarantee the shell's command dispatch rests on: look a command up,
//! spawn the matching entry, thread its exit code into `$?`.
//!
//! The name→entry lookup itself is trivial personality glue (a map) and lives above this; here the
//! entry index is chosen per case, exactly as the shell will compute it. BusyBox-multicall shape
//! (`instantiate_named`, op 11 + `join`, op 1), differential interp==JIT.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites (svm-jit's guard page is unix-only).
#![cfg(unix)]

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const WIN: usize = 128 << 10;
const CARVE: u64 = 64 << 10;

/// One module: parent (func 0) plus three applets — func 1 `true` (→0), func 2 `false` (→1), func 3
/// `echo` (resolve `stdout`, write 3 seeded bytes, →3). The parent seeds `token` into the applet's
/// carve, lays a `stdout` grant record, spawns applet `entry`, joins, and returns its status.
fn src(entry: u64, token: &[u8; 3]) -> String {
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
  ent = i64.const {entry}
  off = i64.const {CARVE}
  sl = i64.const 16
  q = i64.const 0
  vch = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) vinst (gp, gn, ent, off, sl, q)
  r = cap.call 6 1 (i32) -> (i64) vinst (vch)
  return r
}}
func (i64) -> (i64) {{
block0(vt: i64):
  z = i64.const 0
  return z
}}
func (i64) -> (i64) {{
block0(vf: i64):
  o = i64.const 1
  return o
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
        release: svm_run::grant_child_release,
    }
}

fn run_interp(entry: u64, token: &[u8; 3]) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(&src(entry, token)).expect("parse");
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

fn run_jit(entry: u64, token: &[u8; 3]) -> (JitOutcome, Vec<u8>) {
    let m = parse_module(&src(entry, token)).expect("parse");
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

/// Spawning each applet yields its own `(status, stdout)`: `true`→(0,""), `false`→(1,""),
/// `echo`→(3,"hey"). Both backends agree — the shell's dispatch can thread any command's exit code
/// into `$?` and see its output on the inherited stream.
#[test]
fn dispatch_selects_applet_and_threads_its_status() {
    // (entry, expected status, expected stdout)
    let cases: &[(u64, i64, &[u8])] = &[(1, 0, b""), (2, 1, b""), (3, 3, b"hey")];
    for &(entry, status, out) in cases {
        let token = b"hey";
        let (ir, iout) = run_interp(entry, token);
        let (jo, jout) = run_jit(entry, token);
        assert_eq!(
            ir.expect("interp run ok"),
            vec![Value::I64(status)],
            "interp: applet {entry} status"
        );
        assert_eq!(iout, out, "interp: applet {entry} stdout");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[status]),
            "jit: applet {entry} status must be {status}, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: applet {entry} stdout must match interp");
    }
}
