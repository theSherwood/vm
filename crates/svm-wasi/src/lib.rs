//! A minimal **WASI preview1 shim** as an embedder host capability (§7).
//!
//! This is the "host provides a capability; the §7 named-import mechanism carries it" story applied
//! to *real WASI bytes*. svm-wasm transpiles a WASI module, declaring one manifest import
//! `"wasi_snapshot_preview1.<name>"` per WASI function (IMPORTS.md phase 3 — call sites are
//! `call.import <slot>`, the module is never rewritten); [`bind`] grants a single
//! [`svm_interp::cap_id::HOST_FN`] capability and installs one [`BoundImport`] per manifest slot
//! ([`resolve`] maps each name to its op), and [`handler`] implements the WASI ops over the guest
//! window. The WASI *semantics* (the iovec ABI, errno values, the fd table) live **here** — outside
//! both svm-wasm and the interp TCB — exactly the boundary DESIGN.md §7 draws: the binding mechanism
//! is in scope; WASI-the-standard is a host-layer shim a guest reaches only through a granted,
//! masked, type-checked handle.
//!
//! Scope: a deliberately tiny subset — `fd_write` (stdout/stderr) and `proc_exit` — enough for a
//! real "hello world". Not a conformant preview1.
#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex};

use svm_interp::{cap_id, BoundImport, GuestMem, Host, HostFn, Trap};
use svm_ir::ResolvedCap;

/// Op numbers the [`handler`] dispatches on; [`resolve`] maps WASI names to these.
const OP_FD_WRITE: u32 = 0;
const OP_PROC_EXIT: u32 = 1;

/// `__WASI_ERRNO_SUCCESS` / `__WASI_ERRNO_INVAL` — fd_write returns these as its `i32` result.
const ERRNO_SUCCESS: i64 = 0;
const ERRNO_INVAL: i64 = 28;

/// Captured WASI output. The handler appends `fd_write` bytes here so an embedder/test can read what
/// the guest wrote (cheap `Arc` clones share one buffer between the handler and the caller).
#[derive(Clone, Default)]
pub struct WasiOut {
    pub stdout: Arc<Mutex<Vec<u8>>>,
    pub stderr: Arc<Mutex<Vec<u8>>>,
}

/// The §7 import-name resolver for this WASI subset: maps the standard preview1 import names (as
/// svm-wasm declares them, `"<module>.<name>"`) to the [`cap_id::HOST_FN`] capability + op. [`bind`]
/// uses it to build the per-slot bindings; compose with your own policy for other imports. Unknown
/// names return `None`, so binding fails closed.
pub fn resolve(name: &str) -> Option<ResolvedCap> {
    let op = match name {
        "wasi_snapshot_preview1.fd_write" => OP_FD_WRITE,
        "wasi_snapshot_preview1.proc_exit" => OP_PROC_EXIT,
        _ => return None,
    };
    Some(ResolvedCap {
        type_id: cap_id::HOST_FN,
        op,
    })
}

/// Grant a WASI capability handle on `host`, returning the handle and the shared output buffers.
/// Every WASI import in a transpiled module shares this **one** handle, with the op distinguishing
/// the call. Most embedders want [`bind`], which also installs the per-slot import bindings.
pub fn grant(host: &mut Host) -> (i32, WasiOut) {
    let out = WasiOut::default();
    let handle = host.grant_host_fn(handler(out.clone()));
    (handle, out)
}

/// Grant the WASI capability on `host` and install one [`BoundImport`] per manifest slot of `m`
/// (IMPORTS.md phase 3): each import name resolves through [`resolve`] to `(HOST_FN, op)` over the
/// single granted handle. Fails closed (`None`, nothing installed) on a non-WASI import name. The
/// module is never rewritten and the entry takes only its wasm params.
pub fn bind(m: &svm_ir::Module, host: &mut Host) -> Option<WasiOut> {
    let caps: Vec<ResolvedCap> = m
        .imports
        .iter()
        .map(|i| resolve(&i.name))
        .collect::<Option<_>>()?;
    let (handle, out) = grant(host);
    host.set_import_bindings(
        caps.iter()
            .map(|c| BoundImport::required(c.type_id, c.op, handle))
            .collect(),
    );
    Some(out)
}

/// Build the WASI [`HostFn`] handler over `out`. `fd_write` captures into `out`; `proc_exit`
/// terminates the domain with the given code (a non-error [`Trap::Exit`]).
pub fn handler(out: WasiOut) -> HostFn {
    Box::new(move |op, args, mem| match op {
        OP_FD_WRITE => fd_write(&out, args, mem),
        OP_PROC_EXIT => Err(Trap::Exit(args.first().copied().unwrap_or(0) as i32)),
        _ => Err(Trap::CapFault), // unknown WASI op on this handle
    })
}

/// `fd_write(fd, iovs, iovs_len, nwritten) -> errno`: walk the iovec array in the guest window,
/// append each slice to the captured `fd` buffer, write the byte total to `*nwritten`, return 0.
/// Only `fd` 1 (stdout) / 2 (stderr) are supported here (anything else → `EINVAL`).
fn fd_write(out: &WasiOut, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
    let mem = mem.ok_or(Trap::Malformed)?;
    let fd = *args.first().ok_or(Trap::Malformed)? as i32;
    let iovs = *args.get(1).ok_or(Trap::Malformed)? as u64;
    let iovs_len = (*args.get(2).ok_or(Trap::Malformed)?).max(0);
    let nwritten = *args.get(3).ok_or(Trap::Malformed)? as u64;
    let sink = match fd {
        1 => &out.stdout,
        2 => &out.stderr,
        _ => return Ok(vec![ERRNO_INVAL]),
    };
    let mut buf = sink.lock().unwrap_or_else(|e| e.into_inner());
    let mut total: u32 = 0;
    for i in 0..iovs_len as u64 {
        // Each `iovec` is `{ buf: u32, buf_len: u32 }` (8 bytes, little-endian) in the window.
        let entry = mem.read_bytes(iovs + i * 8, 8).ok_or(Trap::Malformed)?;
        let ptr = u32::from_le_bytes(entry[0..4].try_into().unwrap()) as u64;
        let len = u32::from_le_bytes(entry[4..8].try_into().unwrap());
        if len > 0 {
            let data = mem.read_bytes(ptr, len as u64).ok_or(Trap::Malformed)?;
            buf.extend_from_slice(&data);
            total = total.saturating_add(len);
        }
    }
    mem.write_bytes(nwritten, &total.to_le_bytes())
        .ok_or(Trap::Malformed)?;
    Ok(vec![ERRNO_SUCCESS])
}

#[cfg(test)]
mod tests {
    use super::*;
    use svm_interp::run_with_host;

    /// A real WASI preview1 **"hello world"**: imports `wasi_snapshot_preview1.fd_write`, builds an
    /// `iovec` pointing at "hello\n", and writes it to fd 1 — the same shape clang/rustc emit for
    /// `wasm32-wasi` (minimal, hand-written so the test needs no wasi toolchain). svm-wasm transpiles
    /// it (the WASI import → a `CallImport`), [`resolve`] binds the name to the WASI `HostFn`
    /// capability, and the bytes the shim captures prove the whole path carries real WASI bytes.
    const HELLO_WAT: &str = r#"
      (module
        (import "wasi_snapshot_preview1" "fd_write"
          (func $fd_write (param i32 i32 i32 i32) (result i32)))
        (memory 1)
        (data (i32.const 16) "hello\n")
        (func (export "_start")
          (i32.store (i32.const 0) (i32.const 16))   ;; iov.buf      = 16
          (i32.store (i32.const 4) (i32.const 6))    ;; iov.buf_len  = 6
          (drop (call $fd_write
            (i32.const 1)        ;; fd          = stdout
            (i32.const 0)        ;; iovs        = &iov
            (i32.const 1)        ;; iovs_len    = 1
            (i32.const 8)))))    ;; nwritten    -> mem[8]
    "#;

    #[test]
    fn wasi_hello_world() {
        let wasm = wat::parse_str(HELLO_WAT).expect("assemble wat");
        let t = svm_wasm::transpile(&wasm).expect("transpile WASI module");
        // The module now declares one §7 named import.
        assert_eq!(t.module.imports.len(), 1);
        assert_eq!(t.module.imports[0].name, "wasi_snapshot_preview1.fd_write");
        // §7 late binding, phase-3 shape: verify the manifest module once, bind slots at
        // instantiation — no rewrite, no handle args.
        svm_verify::verify_module(&t.module).expect("verify manifest module");
        let mut host = Host::new();
        let out = bind(&t.module, &mut host).expect("bind WASI imports");
        let entry = t
            .exports
            .iter()
            .find(|(n, _)| n == "_start")
            .expect("_start export")
            .1;
        let mut fuel = 10_000_000u64;
        run_with_host(&t.module, entry, &[], &mut fuel, &mut host).expect("interp run");
        assert_eq!(
            &*out.stdout.lock().unwrap(),
            b"hello\n",
            "WASI fd_write reached the captured stdout (interp)"
        );

        // JIT parity: the same resolved module + WASI grant, driven through the production cap thunk.
        // The HostFn capability dispatches through the same `cap_dispatch_slots` the thunk calls, so
        // the demo runs on the production backend unchanged.
        let mut hj = Host::new();
        let jout = bind(&t.module, &mut hj).expect("bind WASI imports");
        let outcome = svm_jit::compile_and_run_with_host(
            &t.module,
            entry,
            &[],
            svm_run::cap_thunk,
            &mut hj as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit compile");
        assert!(
            matches!(outcome, svm_jit::JitOutcome::Returned(_)),
            "jit did not return: {outcome:?}"
        );
        assert_eq!(
            &*jout.stdout.lock().unwrap(),
            b"hello\n",
            "WASI fd_write reached the captured stdout (JIT)"
        );
    }

    /// `proc_exit(code)` terminates the domain with that code (a non-error `Trap::Exit`).
    #[test]
    fn wasi_proc_exit_sets_code() {
        let wat = r#"
          (module
            (import "wasi_snapshot_preview1" "proc_exit" (func $exit (param i32)))
            (memory 1)
            (func (export "_start") (call $exit (i32.const 7))))
        "#;
        let wasm = wat::parse_str(wat).expect("assemble wat");
        let t = svm_wasm::transpile(&wasm).expect("transpile");
        svm_verify::verify_module(&t.module).expect("verify");
        let mut host = Host::new();
        let _out = bind(&t.module, &mut host).expect("bind");
        let entry = t.exports.iter().find(|(n, _)| n == "_start").unwrap().1;
        let mut fuel = 10_000_000u64;
        let r = run_with_host(&t.module, entry, &[], &mut fuel, &mut host);
        assert!(
            matches!(r, Err(Trap::Exit(7))),
            "proc_exit(7) → Trap::Exit(7); got {r:?}"
        );
    }

    /// An unknown WASI import is fail-closed at load (the resolver returns `None`).
    #[test]
    fn unknown_wasi_import_fails_closed() {
        assert!(resolve("wasi_snapshot_preview1.sock_accept").is_none());
    }
}
