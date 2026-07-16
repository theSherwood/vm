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
use svm_run::{cap_thunk, powerbox_resolver};
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
