//! The §7 **general-form** powerbox binding (PROCESS.md S15 (c)): a `_start` that takes **no
//! positional handle parameters** and reaches the real fixed powerbox caps — `Stream` (write) and
//! `Exit` — purely **by name**. [`svm_run::powerbox_resolver`] binds each name to its `(type_id, op)`
//! **plus** the granted handle (`Resolved::CapBound`, patching the import's `ConstI32` placeholder),
//! so the handle is instantiation-time binding, not a positional entry argument threaded from a
//! stashed slot. This is the runner-side foundation the frontend flip (S15 (c2), retiring chibicc's
//! 8-slot `_start`) builds on; here it is proven against hand-authored IR on both backends.
//!
//! Contrast `run.rs::writes_to_stdout_and_returns`, whose entry takes three `i32` handle params and
//! threads `v0` (the stdout handle) into the `cap.call` — the positional powerbox this retires.

use core::ffi::c_void;

use svm_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_run::{cap_thunk, instantiate, powerbox_resolver, Outcome, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// A **paramless** `_start` (`func () -> ()`, no handle arguments) that stores `"ok"` into the window,
/// `write`s it to stdout, then `exit(5)` — both caps reached through named imports whose handle
/// operand is a `ConstI32` **placeholder** (`i32.const 0`) the resolver patches. The `write` import
/// keeps the powerbox `Stream` ABI (`(buf, len)` on the stdout handle — fd is the handle's role); the
/// caps bind by name, not by a positional entry argument.
const NAMED_START: &str = "memory 17\n\
func () -> () {\n\
block0():\n\
  vo = i32.const 111\n\
  va = i64.const 16\n\
  i32.store8 va vo\n\
  vk = i32.const 107\n\
  vb = i64.const 17\n\
  i32.store8 vb vk\n\
  vph = i32.const 0\n\
  vbuf = i64.const 16\n\
  vlen = i64.const 2\n\
  vn = call.import \"write\" (i64, i64) -> (i64) vph (vbuf, vlen)\n\
  vph2 = i32.const 0\n\
  vcode = i32.const 5\n\
  call.import \"exit\" (i32) -> () vph2 (vcode)\n\
  unreachable\n\
}\n";

const WIN: usize = 128 << 10;

/// Grant the fixed §3e powerbox prefix on `host` in slot order and return the handle array
/// `powerbox_resolver` expects (stdout, stdin, exit, memory, addrspace, ioring, blocking, jit).
fn grant_powerbox(host: &mut Host, win: u64) -> [i32; 8] {
    host.set_region_factory(svm_run::new_shared_region);
    host.set_jit_validator(svm_run::jit_blob_validator);
    let mem_log2 = (win != 0).then(|| win.trailing_zeros() as u8);
    [
        host.grant_stream(StreamRole::Out),
        host.grant_stream(StreamRole::In),
        host.grant_exit(),
        host.grant_memory(),
        host.grant_address_space(0, win),
        host.grant_io_ring(),
        host.grant_blocking(std::time::Duration::ZERO, None),
        host.grant_jit(mem_log2),
    ]
}

#[test]
fn paramless_start_binds_stream_and_exit_by_name_on_both_backends() {
    let raw = parse_module(NAMED_START).expect("parse");
    assert!(
        raw.funcs[0].params.is_empty(),
        "the entry takes no positional handle parameters"
    );

    // Grant before resolve (the §7 instantiation ordering), identically on two hosts; deterministic
    // grant order gives both backends the same handles, so one resolved module serves both.
    let mut ih = Host::new();
    let ihandles = grant_powerbox(&mut ih, WIN as u64);
    let mut jh = Host::new();
    let jhandles = grant_powerbox(&mut jh, WIN as u64);
    assert_eq!(
        ihandles, jhandles,
        "identical grant order → identical handles"
    );

    let m = svm_ir::resolve_imports_with(&raw, powerbox_resolver(ihandles))
        .expect("powerbox names resolve to (type_id, op, handle)");
    assert!(m.imports.is_empty(), "resolution is import-free");
    verify_module(&m).expect("verify the resolved module");

    // Interpreter — the paramless entry takes no args; authority came in at resolve.
    let mut fuel = 5_000_000u64;
    let ir = run_capture_reserved_with_host(&m, 0, &[], &mut fuel, &[0u8; WIN], 0, &mut ih).0;
    assert_eq!(
        ir,
        Err(Trap::Exit(5)),
        "interp: exit(5) via the named Exit cap"
    );
    assert_eq!(ih.stdout, b"ok", "interp: write via the named Stream cap");

    // JIT parity — the same CapBound handles, so identical stdout + exit code.
    let jo = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[],
        &[0u8; WIN],
        0,
        cap_thunk,
        &mut jh as *mut Host as *mut c_void,
    )
    .expect("jit compiles")
    .0;
    assert!(
        matches!(jo, JitOutcome::Exited(5)),
        "jit: exit(5) must match interp, got {jo:?}"
    );
    assert_eq!(jh.stdout, ih.stdout, "jit: stdout must match interp");
}

/// The **named-export** powerbox path end to end through the high-level embedding API
/// (`instantiate` → `call`): a paramless `_start` marked `export "_start" 0` obtains its `stdout`
/// handle **by name** (`cap.self.resolve("stdout")`) — no positional entry argument, no slot index —
/// and writes through it. The runner sees the named-export marker, grants the fixed powerbox, and
/// registers each cap under its name (F7) so the resolve succeeds; `call` runs it on both backends
/// and asserts they agree. This is the runtime shape the chibicc `_start` flip (S15 (c2)) produces.
const NAMED_EXPORT_START: &str = "memory 17\n\
export \"_start\" 0\n\
func () -> (i32) {\n\
block0():\n\
  p0 = i64.const 64\n\
  s = i32.const 115\n\
  i32.store8 p0 s\n\
  p1 = i64.const 65\n\
  t0 = i32.const 116\n\
  i32.store8 p1 t0\n\
  p2 = i64.const 66\n\
  d = i32.const 100\n\
  i32.store8 p2 d\n\
  p3 = i64.const 67\n\
  o0 = i32.const 111\n\
  i32.store8 p3 o0\n\
  p4 = i64.const 68\n\
  u = i32.const 117\n\
  i32.store8 p4 u\n\
  p5 = i64.const 69\n\
  t1 = i32.const 116\n\
  i32.store8 p5 t1\n\
  nptr = i64.const 64\n\
  nlen = i64.const 6\n\
  vh = cap.self.resolve nptr nlen\n\
  vo = i32.const 111\n\
  a = i64.const 16\n\
  i32.store8 a vo\n\
  vk = i32.const 107\n\
  b = i64.const 17\n\
  i32.store8 b vk\n\
  vbuf = i64.const 16\n\
  vlen = i64.const 2\n\
  vn = cap.call 0 1 (i64, i64) -> (i64) vh (vbuf, vlen)\n\
  vr = i32.const 0\n\
  return vr\n\
}\n";

#[test]
fn named_export_start_resolves_stdout_by_name_and_runs_both_backends() {
    let m = parse_module(NAMED_EXPORT_START).expect("parse");
    assert!(
        svm_run::is_named_powerbox_entry(&m),
        "export \"_start\" 0 + paramless func 0 marks a named-export powerbox entry"
    );
    let inst = instantiate(m).expect("instantiate");
    // `call` grants the fixed powerbox + registers cap names, runs `_start` on both backends, and
    // asserts they agree — the resolve-by-name succeeds because the runner registered "stdout".
    let run = inst.call("_start", &[]).expect("run both backends");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(0)]));
    assert_eq!(
        run.stdout, b"ok",
        "wrote via the name-resolved stdout handle"
    );
}

/// A `SharedRegion` op (`vm_region_map`) is **not** a powerbox slot — its region handle is minted at
/// runtime by `vm_region_create`, so it stays a call-site operand. [`powerbox_resolver`] returns
/// `None` for it, so a program using regions composes the powerbox resolver with a runtime-handle
/// one; here we just pin that the powerbox resolver declines it (fail-closed, not a wrong binding).
#[test]
fn powerbox_resolver_declines_runtime_handle_ops() {
    let handles = [10, 11, 12, 13, 14, 15, 16, 17];
    let r = powerbox_resolver(handles);
    assert!(r("write").is_some(), "a powerbox cap binds by name");
    assert!(r("exit").is_some(), "exit binds by name");
    assert!(r("vm_map").is_some(), "the Memory cap binds by name");
    assert!(
        r("vm_region_map").is_none(),
        "a runtime-minted region handle is not a powerbox slot"
    );
    assert!(r("nonesuch").is_none(), "an unknown name fails closed");
}
