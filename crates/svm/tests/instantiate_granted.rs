//! PROCESS.md S2 — `Instantiator.instantiate_granted` (op 8): a parent re-grants one of its own
//! coordinate-free capabilities (`Stream`/`Exit`/`Clock`) into a §14 child's powerbox, so the child
//! is **not born destitute** — it can do I/O. This is the load-bearing "children can hold
//! capabilities" primitive the process substrate needs (a shell hands its child stdout/stderr/stdin).
//!
//! `instantiate_granted(grant_handle, entry, off, size_log2, quota) -> child | -EINVAL` is exactly
//! `instantiate` (op 0) plus a leading handle that is re-granted into the child; the child receives it
//! as its **third** entry arg (after `Instantiator`, `AddressSpace`). A forged / non-copyable handle
//! (an index-carrying or window-coordinate cap) is a `CapFault`. Interpreter-first (the JIT
//! Instantiator powerbox is a baked thunk — a follow-up, like the base Instantiator's own JIT port).

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// func 0 (parent, `(Instantiator, grant_handle)`): `instantiate_granted` the child (func 1) in a
/// 4 KiB carve at offset 0, re-granting the parent's `grant_handle`, then `join` and return the
/// child's result.
///
/// func 1 (child, `(Instantiator, AddressSpace, Stream)`): write the three bytes `"hi\n"` into its own
/// window and `Stream.write(0, 3)` through the **inherited** handle, then return 7.
const SRC: &str = "memory 17\n\
func (i32, i32) -> (i64) {\n\
block0(vinst: i32, vstream: i32):\n\
  vgh = i64.extend_i32_u vstream\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 8 (i64, i64, i64, i64, i64) -> (i32) vinst (vgh, ventry, voff, vsl, vq)\n\
  vres = cap.call 6 1 (i32) -> (i64) vinst (vch)\n\
  return vres\n\
}\n\
func (i64, i64, i64) -> (i64) {\n\
block0(vcinst: i64, vcas: i64, vcstream: i64):\n\
  v0 = i64.const 0\n\
  vhb = i32.const 104\n\
  i32.store8 v0 vhb\n\
  v1 = i64.const 1\n\
  vib = i32.const 105\n\
  i32.store8 v1 vib\n\
  v2 = i64.const 2\n\
  vnb = i32.const 10\n\
  i32.store8 v2 vnb\n\
  vsh = i32.wrap_i64 vcstream\n\
  vptr = i64.const 0\n\
  vlen = i64.const 3\n\
  vw = cap.call 0 1 (i64, i64) -> (i64) vsh (vptr, vlen)\n\
  v7 = i64.const 7\n\
  return v7\n\
}\n";

fn run(inst_first: bool) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let sh = host.grant_stream(StreamRole::Out);
    // The parent entry is `(Instantiator, grant_handle)`. The happy path passes the Stream as the
    // grant; the negative path passes the Instantiator itself (a window-coordinate cap → not
    // copyable) to prove it is refused.
    let grant = if inst_first { ih } else { sh };
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(grant)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    );
    // The child's stdout was shared into the parent host's sink (stdio inheritance), so read the
    // effective bytes, not the now-promoted local `stdout` Vec.
    (res, host.stdout_bytes())
}

#[test]
fn child_writes_stdout_through_inherited_stream() {
    let (res, out) = run(false); // grant the Stream
    assert_eq!(res, Ok(vec![Value::I64(7)]), "child ran and joined");
    assert_eq!(
        out, b"hi\n",
        "the child produced output through the re-granted stdout Stream"
    );
}

#[test]
fn non_copyable_grant_is_capfault() {
    // Passing the Instantiator handle as the grant: a window-coordinate cap is not re-grantable, so
    // `resolve_copyable` refuses it and the `instantiate_granted` cap.call is a `CapFault`.
    let (res, out) = run(true);
    assert_eq!(
        res,
        Err(Trap::CapFault),
        "a non-copyable grant must fault, not silently succeed"
    );
    assert!(out.is_empty(), "nothing should have been written");
}
