//! `svm-run` — the **embedding runtime**: instantiate a verified module with the MVP powerbox
//! (§3e) and run it on the Cranelift JIT, returning its outcome and the bytes it wrote.
//!
//! This is the single, reusable host glue — the `cap.call` trampoline ([`cap_thunk`]) plus the
//! powerbox grant ([`run_powerbox`]) — that was previously copy-pasted across the JIT test
//! harnesses (`c_frontend.rs`, `jit_diff.rs`). The `svm-run` **CLI** is a thin wrapper over it.
//!
//! It is *not* escape-TCB: the verifier (run before this) is what makes a module safe to run;
//! this crate only wires the host capabilities a guest is granted. A guest that traps
//! (out-of-window fault, `unreachable`, …) is **detect-and-killed** (§5) — surfaced here as an
//! `Err`, never undefined behaviour in the host.

use core::ffi::c_void;

use svm_interp::{GuestMem, Host, StreamRole, Trap, WindowMem};
use svm_ir::{Module, ValType};

// Re-export the value type so embedders (and the CLI) need not also depend on `svm-interp`.
pub use svm_interp::Value;
use svm_jit::{compile_and_run, compile_and_run_with_host, JitOutcome, TrapKind, EXIT_CODE};

/// The host trampoline bridging the JIT's [`svm_jit::CapThunk`] ABI (§9) to the reference
/// [`Host`]'s capability dispatch — the host code a real embedder supplies. One shared copy.
///
/// # Safety
/// Honours the `CapThunk` contract: `ctx` is a live `*mut Host`; `args`/`results` are valid for
/// `n_args`/`n_results`; `mem_base` (when non-null) is valid for `mem_size`; `trap_out` is
/// writable. The trap cell is encoded as the JIT expects: `0` = ok, a [`TrapKind`] for a fault,
/// or `EXIT_CODE | (code << 32)` for an `Exit`.
pub unsafe extern "C" fn cap_thunk(
    ctx: *mut c_void,
    mem_base: *mut u8,
    mem_size: u64,
    type_id: u32,
    op: u32,
    handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    let host = &mut *(ctx as *mut Host);
    let arg_slots = std::slice::from_raw_parts(args, n_args as usize);
    let mut empty: [u8; 0] = [];
    let window: &mut [u8] = if mem_base.is_null() {
        &mut empty
    } else {
        std::slice::from_raw_parts_mut(mem_base, mem_size as usize)
    };
    let mut wm = WindowMem::new(window, mem_size);
    let gm: Option<&mut dyn GuestMem> = if mem_base.is_null() {
        None
    } else {
        Some(&mut wm)
    };
    match host.cap_dispatch_slots(type_id, op, handle, arg_slots, gm) {
        Ok(res) => {
            let out = std::slice::from_raw_parts_mut(results, n_results as usize);
            for (o, r) in out.iter_mut().zip(res) {
                *o = r;
            }
            *trap_out = 0;
        }
        Err(Trap::Exit(code)) => *trap_out = EXIT_CODE as i64 | ((code as i64) << 32),
        Err(_) => *trap_out = TrapKind::CapFault as i64,
    }
}

/// How a guest program ended: its entry returned values, or it invoked `Exit(code)` (§3e).
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Returned(Vec<Value>),
    Exited(i32),
}

/// The result of running a program through the powerbox: how it ended, plus the bytes it wrote
/// to stdout/stderr via the `Stream` capabilities.
#[derive(Debug, Clone)]
pub struct Run {
    pub outcome: Outcome,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// The frontend's `_start(stdout, stdin, exit)` powerbox entry shape — three `i32` capability
/// handles (function 0). A module whose entry matches this is a runnable *program*; anything
/// else is a bare kernel (run with [`run_kernel`]).
pub fn is_powerbox_entry(module: &Module) -> bool {
    matches!(
        module.funcs.first().map(|f| f.params.as_slice()),
        Some([ValType::I32, ValType::I32, ValType::I32])
    )
}

fn typed(t: ValType, v: i64) -> Value {
    match t {
        ValType::I32 => Value::I32(v as i32),
        ValType::I64 => Value::I64(v),
        ValType::F32 => Value::F32(f32::from_bits(v as u32)),
        ValType::F64 => Value::F64(f64::from_bits(v as u64)),
    }
}

/// Run `module`'s entry (function 0) on the JIT under the MVP powerbox (§3e): a writable
/// `stdout`, a readable `stdin` seeded from `stdin`, and `Exit` — the three handles the
/// frontend's `_start` expects, granted in declared order. Returns the outcome and captured
/// output. `Err` if the (already-verified) module fails to JIT-compile, or if the guest
/// **traps** (detect-and-kill, §5) — the guest can never corrupt the host.
pub fn run_powerbox(module: &Module, stdin: &[u8]) -> Result<Run, String> {
    let mut host = Host::new();
    host.stdin = stdin.to_vec();
    // Grant in the powerbox's declared import order: stdout, stdin, exit (§3e / D44).
    let slots = [
        host.grant_stream(StreamRole::Out) as i64,
        host.grant_stream(StreamRole::In) as i64,
        host.grant_exit() as i64,
    ];
    let jit = compile_and_run_with_host(
        module,
        0,
        &slots,
        cap_thunk,
        &mut host as *mut Host as *mut c_void,
    )
    .map_err(|e| format!("JIT compile failed: {e:?}"))?;

    let outcome = match jit {
        JitOutcome::Returned(s) => {
            let results = &module.funcs[0].results;
            Outcome::Returned(s.iter().zip(results).map(|(&v, t)| typed(*t, v)).collect())
        }
        JitOutcome::Exited(code) => Outcome::Exited(code),
        JitOutcome::Trapped(kind) => {
            return Err(format!("guest trapped ({kind:?}) — detect-and-kill (§5)"))
        }
    };
    Ok(Run {
        outcome,
        stdout: host.stdout,
        stderr: host.stderr,
    })
}

/// Run a bare (non-powerbox) kernel — `module`'s entry on the JIT with `args` and no host
/// capabilities — returning its typed result values. For hand-written IR that is a pure
/// function rather than a program (e.g. the benchmark kernels). `Err` on compile failure,
/// a guest trap, or an `Exit` (a kernel should not call one).
pub fn run_kernel(module: &Module, args: &[i64]) -> Result<Vec<Value>, String> {
    match compile_and_run(module, 0, args).map_err(|e| format!("JIT compile failed: {e:?}"))? {
        JitOutcome::Returned(s) => {
            let results = &module.funcs[0].results;
            Ok(s.iter().zip(results).map(|(&v, t)| typed(*t, v)).collect())
        }
        JitOutcome::Exited(code) => Err(format!("kernel called Exit({code})")),
        JitOutcome::Trapped(kind) => Err(format!("kernel trapped ({kind:?})")),
    }
}
