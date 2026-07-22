//! PROCESS.md S2 (JIT parity) — `Instantiator.instantiate_named` (op 11) on the **JIT**: a **multi-cap
//! grant list** re-granted into a §14 child **by name**, discovered with `cap.self.resolve` (the
//! general form of op 8's single positional grant). The interpreter has done this since S2
//! (`instantiate_named.rs`); this pins the JIT to the same observable behavior.
//!
//! The JIT keeps the `Host` opaque, so the child powerbox is built host-side by
//! `svm_run::grant_named_child_build`: it reads the guest's `grants_n` 16-byte records
//! `{name_off, name_len, handle, flags}` from the parent window and re-grants each copyable handle
//! under its name (via the shared `Host::spawn_named_child`), so both backends register the same names
//! against the same shared sinks. The child then finds each cap by `cap.self.resolve` — which lowers to
//! the run's `cap.call` thunk with the child host as ctx, so name resolution works unchanged.

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

/// func 0 (parent, `(Instantiator, stdout_handle, stderr_handle)`): lay out two grant records at
/// window 0/16 with names `"stdout"`@100 and `"stderr"`@110, `instantiate_named` a 64 KiB child at
/// offset 64 KiB granting both, `join`, return the child's result.
///
/// func 1 (child, `(Instantiator)`): resolve `"stdout"` and `"stderr"` by name (each written into its
/// own window first), write `'O'` to the former and `'E'` to the latter, return 7. (Byte-identical to
/// the interpreter's `instantiate_named.rs` source, so the two backends run the same program.)
const SRC: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vout: i32, verr: i32) {
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
  a16 = i64.const 16
  n110 = i32.const 110
  i32.store a16 n110
  a20 = i64.const 20
  i32.store a20 n6
  a24 = i64.const 24
  i32.store a24 verr
  a28 = i64.const 28
  i32.store a28 z0
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  ce = i32.const 101
  cr = i32.const 114
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
  p110 = i64.const 110
  i32.store8 p110 cs
  p111 = i64.const 111
  i32.store8 p111 ct
  p112 = i64.const 112
  i32.store8 p112 cd
  p113 = i64.const 113
  i32.store8 p113 ce
  p114 = i64.const 114
  i32.store8 p114 cr
  p115 = i64.const 115
  i32.store8 p115 cr
  gp = i64.const 0
  gn = i64.const 2
  ent = i64.const 1
  off = i64.const 65536
  sl = i64.const 16
  q = i64.const 0
  vch = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) vinst (gp, gn, ent, off, sl, q)
  r = cap.call 6 1 (i32) -> (i64) vinst (vch)
  return r
  }
}
func (i64) -> (i64) {
block 0 (vci: i64) {
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  ce = i32.const 101
  cr = i32.const 114
  a0 = i64.const 0
  i32.store8 a0 cs
  a1 = i64.const 1
  i32.store8 a1 ct
  a2 = i64.const 2
  i32.store8 a2 cd
  a3 = i64.const 3
  i32.store8 a3 co
  a4 = i64.const 4
  i32.store8 a4 cu
  a5 = i64.const 5
  i32.store8 a5 ct
  len6 = i64.const 6
  hout = cap.self.resolve a0 len6
  a16 = i64.const 16
  cO = i32.const 79
  i32.store8 a16 cO
  one = i64.const 1
  wo = cap.call 0 1 (i64, i64) -> (i64) hout (a16, one)
  a32 = i64.const 32
  i32.store8 a32 cs
  a33 = i64.const 33
  i32.store8 a33 ct
  a34 = i64.const 34
  i32.store8 a34 cd
  a35 = i64.const 35
  i32.store8 a35 ce
  a36 = i64.const 36
  i32.store8 a36 cr
  a37 = i64.const 37
  i32.store8 a37 cr
  herr = cap.self.resolve a32 len6
  a40 = i64.const 40
  cE = i32.const 69
  i32.store8 a40 cE
  we = cap.call 0 1 (i64, i64) -> (i64) herr (a40, one)
  v7 = i64.const 7
  return v7
  }
}
"#;

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
    }
}

/// (parent result, stdout bytes, stderr bytes) on the interpreter.
fn run_interp() -> (Result<Vec<Value>, svm_interp::Trap>, Vec<u8>, Vec<u8>) {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let oh = host.grant_stream(StreamRole::Out);
    let eh = host.grant_stream(StreamRole::Err);
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(oh), Value::I32(eh)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    );
    (res, host.stdout_bytes(), host.stderr_bytes())
}

/// (parent outcome, stdout bytes, stderr bytes) on the JIT with the named-grant hooks installed.
fn run_jit() -> (JitOutcome, Vec<u8>, Vec<u8>) {
    let m = parse_module(SRC).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let oh = host.grant_stream(StreamRole::Out);
    let eh = host.grant_stream(StreamRole::Err);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[ih as i64, oh as i64, eh as i64],
        &[0u8; 128 << 10],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        Some(grant_hooks()),
    )
    .expect("jit");
    (jo, host.stdout_bytes(), host.stderr_bytes())
}

#[test]
fn named_child_resolves_two_grants_matches_interp() {
    let (ir, iout, ierr) = run_interp();
    let (jo, jout, jerr) = run_jit();
    // Interpreter reference: the child resolved both names and wrote one byte to each.
    assert_eq!(ir, Ok(vec![Value::I64(7)]), "interp: child ran and joined");
    assert_eq!(iout, b"O", "interp: name-resolved stdout");
    assert_eq!(ierr, b"E", "interp: name-resolved stderr");
    // JIT parity: same return value, same bytes into the same (shared) sinks.
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[7]),
        "jit: named child must join with 7, got {jo:?}"
    );
    assert_eq!(jout, iout, "jit: name-resolved stdout must match interp");
    assert_eq!(jerr, ierr, "jit: name-resolved stderr must match interp");
}
