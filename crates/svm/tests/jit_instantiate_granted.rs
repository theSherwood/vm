//! PROCESS.md S2 (JIT parity) — `Instantiator.instantiate_granted` (op 8) on the **JIT**. A parent
//! re-grants one of its own coordinate-free capabilities (`Stream`/`Exit`/`Clock`) into a §14 child's
//! powerbox, so the child is **not born destitute** — it can do I/O. The interpreter has done this
//! since S2 (`instantiate_granted.rs`); this pins the JIT to the same observable behavior.
//!
//! The JIT keeps the `Host` opaque, so the child powerbox is built host-side by
//! `svm_run::grant_child_build` (freed by `grant_child_release`) — the same `Host::spawn_granted_child`
//! the interpreter's own op-8 path uses, so both backends hand the child an identical set of handles
//! and share the parent's stdout sink (stdio inheritance). A granted JIT child still runs synchronously
//! at `instantiate` and gets no nesting `InstEnv` (recursive nesting of a *granted* child is a
//! follow-up, tied to JIT async children, S1c) — but a child that does I/O and returns is exactly the
//! "born with stdout" case, and it matches the interpreter byte-for-byte.

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use svm_jit::{
    compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome, TrapKind,
};
use svm_text::parse_module;
use svm_verify::verify_module;

/// func 0 (parent, `(Instantiator, grant_handle)`): `instantiate_granted` the child (func 1) in a
/// 4 KiB carve at offset 0, re-granting the parent's `grant_handle`, then `join` and return the
/// child's result.
///
/// func 1 (child, `(Instantiator, AddressSpace, Stream)`): write the three bytes `"hi\n"` into its own
/// window and `Stream.write(0, 3)` through the **inherited** handle, then return 7. (Byte-identical to
/// the interpreter's `instantiate_granted.rs` source, so the two backends run the same program.)
const SRC: &str = "memory 17\n\
func (i32, i32) -> (i64) {\n\
block 0 (vinst: i32, vstream: i32) {\n\
  vgh = i64.extend_i32_u vstream\n\
  ventry = i64.const 1\n\
  voff = i64.const 0\n\
  vsl = i64.const 12\n\
  vq = i64.const 0\n\
  vch = cap.call 6 8 (i64, i64, i64, i64, i64) -> (i32) vinst (vgh, ventry, voff, vsl, vq)\n\
  vres = cap.call 6 1 (i32) -> (i64) vinst (vch)\n\
  return vres\n\
  }\n\
}\n\
func (i64, i64, i64) -> (i64) {\n\
block 0 (vcinst: i64, vcas: i64, vcstream: i64) {\n\
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
  }\n\
}\n";

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

/// Run `SRC` on the interpreter. `stream_grant` picks the parent's second arg: the re-grantable
/// `Stream` (happy path) or the non-copyable `Instantiator` (negative path). Returns the parent
/// result and the effective stdout bytes (the child's output, shared into the parent's sink).
fn run_interp(stream_grant: bool) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let sh = host.grant_stream(StreamRole::Out);
    let grant = if stream_grant { sh } else { ih };
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
    (res, host.stdout_bytes())
}

/// Run `SRC` on the JIT with the granted-child host callbacks installed. Same shape as `run_interp`.
fn run_jit(stream_grant: bool) -> (JitOutcome, Vec<u8>) {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let sh = host.grant_stream(StreamRole::Out);
    let grant = if stream_grant { sh } else { ih };
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[ih as i64, grant as i64],
        &[0u8; 128 << 10],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        Some(grant_hooks()),
    )
    .expect("jit");
    (jo, host.stdout_bytes())
}

#[test]
fn granted_child_writes_stdout_matches_interp() {
    let (ir, iout) = run_interp(true);
    let (jo, jout) = run_jit(true);
    // Interpreter reference: the child ran, wrote through the re-granted stdout, joined with 7.
    assert_eq!(ir, Ok(vec![Value::I64(7)]), "interp: child ran and joined");
    assert_eq!(
        iout, b"hi\n",
        "interp: child output through re-granted stdout"
    );
    // JIT parity: same return value, same bytes into the same (shared) sink.
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[7]),
        "jit: granted child must join with 7, got {jo:?}"
    );
    assert_eq!(jout, iout, "jit: granted child's stdout must match interp");
}

#[test]
fn non_copyable_grant_capfaults_on_both() {
    // Granting the Instantiator handle (a window-coordinate cap) is refused by `resolve_copyable`, so
    // `instantiate_granted` is a `CapFault` on both backends — never a silent success.
    let (ir, iout) = run_interp(false);
    let (jo, jout) = run_jit(false);
    assert_eq!(ir, Err(Trap::CapFault), "interp: non-copyable grant faults");
    assert!(
        matches!(jo, JitOutcome::Trapped(TrapKind::CapFault)),
        "jit: non-copyable grant must fault, got {jo:?}"
    );
    assert!(iout.is_empty(), "interp: nothing written");
    assert!(jout.is_empty(), "jit: nothing written");
}
