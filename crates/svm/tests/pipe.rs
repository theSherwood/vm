//! PROCESS.md §4 / S4 — a **host-served pipe**: `Host::grant_pipe()` mints one FIFO and grants both
//! ends as `Stream`-typed handles (a pipe end *is* a stream — read/write/close). Bytes written to the
//! write end are drained, FIFO order, by the read end. The personality's intra-domain byte IPC (a shell
//! wiring `cmd1 | cmd2` hands each side one end).
//!
//! A pipe end dispatches through the same generic `cap.call` → `Host` path as any `Stream`, so the
//! interpreter and JIT service it identically — these run each program on both backends and assert
//! equal results (parity for free, like `Budget`/`Clock`).

use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

fn run_interp(src: &str, host: &mut Host, w: i32, r: i32) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(w), Value::I32(r)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        host,
    )
    .0
}

fn run_jit(src: &str, host: &mut Host, w: i32, r: i32) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[w as i64, r as i64],
        &[0u8; 128 << 10],
        0,
        svm_run::cap_thunk,
        host as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit")
    .0
}

fn both(src: &str) -> (Result<Vec<Value>, svm_interp::Trap>, JitOutcome) {
    let mut ih = Host::new();
    let (iw, ir_) = ih.grant_pipe();
    let ir = run_interp(src, &mut ih, iw, ir_);
    let mut jh = Host::new();
    let (jw, jr) = jh.grant_pipe();
    let jo = run_jit(src, &mut jh, jw, jr);
    (ir, jo)
}

/// func 0 `(write_end, read_end)`: write `"hi"` to the write end, read 2 bytes back from the read end,
/// then encode `read_count * 65536 + byte0 * 256 + byte1`. `'h'=104`, `'i'=105`, count `2` →
/// `2*65536 + 104*256 + 105` = `157801` — proving the bytes flowed through the FIFO in order.
const WRITE_THEN_READ: &str = "memory 17\n\
func (i32, i32) -> (i64) {\n\
block0(vw: i32, vr: i32):\n\
  a0 = i64.const 0\n\
  ch = i32.const 104\n\
  i32.store8 a0 ch\n\
  a1 = i64.const 1\n\
  ci = i32.const 105\n\
  i32.store8 a1 ci\n\
  vlen = i64.const 2\n\
  vn = cap.call 0 1 (i64, i64) -> (i64) vw (a0, vlen)\n\
  a16 = i64.const 16\n\
  vread = cap.call 0 0 (i64, i64) -> (i64) vr (a16, vlen)\n\
  vb0 = i32.load8_u a16\n\
  a17 = i64.const 17\n\
  vb1 = i32.load8_u a17\n\
  k256 = i32.const 256\n\
  k65536 = i32.const 65536\n\
  vrd = i32.wrap_i64 vread\n\
  t0 = i32.mul vrd k65536\n\
  t1 = i32.mul vb0 k256\n\
  t2 = i32.add t0 t1\n\
  t3 = i32.add t2 vb1\n\
  vresult = i64.extend_i32_u t3\n\
  return vresult\n\
}\n";

/// func 0 `(write_end, read_end)`: read from the pipe **before** anything is written (non-blocking →
/// `0`), then try to **write to the read end** (wrong direction → `-EINVAL` = `-22`). Encode
/// `empty_read * 1000 + (0 - bad_write)` = `0*1000 + 22` = `22`.
const EMPTY_AND_WRONG_DIRECTION: &str = "memory 17\n\
func (i32, i32) -> (i64) {\n\
block0(vw: i32, vr: i32):\n\
  a0 = i64.const 0\n\
  vlen = i64.const 4\n\
  vempty = cap.call 0 0 (i64, i64) -> (i64) vr (a0, vlen)\n\
  vbad = cap.call 0 1 (i64, i64) -> (i64) vr (a0, vlen)\n\
  vzero = i64.const 0\n\
  vd = i64.sub vzero vbad\n\
  k1000 = i64.const 1000\n\
  vt = i64.mul vempty k1000\n\
  vsum = i64.add vt vd\n\
  return vsum\n\
}\n";

#[test]
fn write_then_read_flows_through_fifo_on_both() {
    let (ir, jo) = both(WRITE_THEN_READ);
    assert_eq!(
        ir,
        Ok(vec![Value::I64(157_801)]),
        "interp: read 2 bytes 'h','i' back through the pipe"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[157_801]),
        "jit: pipe read must match interp, got {jo:?}"
    );
}

#[test]
fn empty_read_and_wrong_direction_on_both() {
    let (ir, jo) = both(EMPTY_AND_WRONG_DIRECTION);
    assert_eq!(
        ir,
        Ok(vec![Value::I64(22)]),
        "interp: empty read is 0; writing the read end is -EINVAL"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[22]),
        "jit: must match interp, got {jo:?}"
    );
}
