//! **Blocking-stdin park** on the resumable [`bytecode::Vcpu`] (the persistent-session seam — the browser
//! Postgres console). With [`Host::set_stdin_blocking`] on, a `read` on an exhausted stdin buffer must
//! *suspend* the vCPU ([`VcpuEvent::StdinPark`]) instead of returning EOF — and resume, re-issuing the
//! same read, once the host pushes more bytes. This proves the park/resume round-trip end to end: no
//! bytes lost, the read re-executes, and the guest never sees a spurious EOF.

use svm_interp::bytecode::{self, VcpuEvent};
use svm_interp::{Host, StreamRole, Value};
use svm_text::parse_module;

// A guest that forever: read(stdin, buf, 64) → write(stdout, buf, n) → repeat. `_start(vin, vout)`
// takes the two stream handles positionally. Reading an empty stdin under blocking mode parks it.
const ECHO: &str = r#"memory 16
func (i32, i32) -> (i64) {
block 0 (vin: i32, vout: i32) {
  br 1(vin, vout)
}
block 1 (vin1: i32, vout1: i32) {
  vptr = i64.const 1024
  vlen = i64.const 64
  vn = cap.call 0 0 (i64, i64) -> (i64) vin1 (vptr, vlen)
  vw = cap.call 0 1 (i64, i64) -> (i64) vout1 (vptr, vn)
  br 1(vin1, vout1)
  }
}
"#;

#[test]
fn stdin_read_parks_and_resumes() {
    let m = parse_module(ECHO).expect("parse echo guest");
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile echo guest");

    let mut host = Host::new();
    let vin = host.grant_stream(StreamRole::In);
    let vout = host.grant_stream(StreamRole::Out);
    host.set_stdin_blocking(true);
    host.push_stdin(b"abc"); // preload the first chunk

    let slots = [Value::I32(vin), Value::I32(vout)];
    // Engine-backed reservation, the resumable twin of the one-shot reserved run (small reserve for the test).
    let mut vcpu = bytecode::Vcpu::new_root_reserved_with_powerbox(&prog, 0, &slots, &[], host, 20)
        .expect("build root vcpu");

    // First run: echoes "abc", then the next read finds the buffer empty and parks (not EOF/exit).
    assert!(
        matches!(vcpu.run(), VcpuEvent::StdinPark),
        "an exhausted blocking stdin must park the vCPU, not return EOF",
    );
    assert_eq!(
        vcpu.host_mut().stdout,
        b"abc",
        "the preloaded chunk was echoed"
    );

    // Push more and resume: the parked read re-issues, sees "de", echoes it, parks again — nothing lost.
    vcpu.push_stdin(b"de");
    assert!(matches!(vcpu.run(), VcpuEvent::StdinPark));
    assert_eq!(
        vcpu.host_mut().stdout,
        b"abcde",
        "resumed read appended the new bytes"
    );

    // A second push/resume proves the loop keeps parking (a real REPL parks after every prompt).
    vcpu.push_stdin(b"f");
    assert!(matches!(vcpu.run(), VcpuEvent::StdinPark));
    assert_eq!(vcpu.host_mut().stdout, b"abcdef");
}

// Sanity: with blocking OFF (the default — the one-shot `svm_run_pg` / oracle path), an exhausted read
// returns EOF and the guest runs to completion, unchanged. Here the guest reads once and returns `n`.
const READ_ONCE: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vin: i32) {
  vptr = i64.const 1024
  vlen = i64.const 64
  vn = cap.call 0 0 (i64, i64) -> (i64) vin (vptr, vlen)
  return vn
  }
}
"#;

#[test]
fn non_blocking_stdin_returns_eof_at_end() {
    let m = parse_module(READ_ONCE).expect("parse read-once guest");
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let mut host = Host::new();
    let vin = host.grant_stream(StreamRole::In);
    // stdin_block defaults to false; buffer is empty → read returns 0 (EOF), the guest returns 0.
    let slots = [Value::I32(vin)];
    let mut vcpu = bytecode::Vcpu::new_root_reserved_with_powerbox(&prog, 0, &slots, &[], host, 20)
        .expect("build vcpu");
    match vcpu.run() {
        VcpuEvent::Done(vals) => {
            assert_eq!(vals.first(), Some(&Value::I64(0)), "empty read = EOF = 0")
        }
        _ => panic!("expected Done(0) with blocking off (empty read should be EOF, not a park)"),
    }
}
