//! The **`exec` capability** (EXEC.md): shell-out as granted, attenuated authority. One
//! interface a guest resolves by name (`"exec"`); *what a spawn is* — a real host process, a
//! scripted table entry, or (later) a child svm domain — is the wirer's choice, invisible to
//! the guest. This module holds the real-subprocess backend, [`host_exec`]: the capability
//! *is* the program allowlist, the way [`crate::fs::host_fs`]'s capability is the rooted
//! directory. The deterministic [`scripted_exec`] backend and the shared wire protocol live in
//! the wasm-safe `svm-exec` crate, re-exported here so `svm_run::exec::*` is one surface.
//!
//! No subprocess authority exists un-granted: without a grant, `cap.self.resolve("exec")` is
//! negative and the guest's fallback runs. An allowlist miss is a refused op (`-EPERM`), never
//! a trap — the same failure shape as a `scripted_exec` table miss, so a guest cannot tell the
//! backends apart by how they refuse (interposition invisibility, per EXEC.md).

pub use svm_exec::*;

use crate::{Backend, HostCap, Instance, Limits, Outcome, RunConfig};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;
use svm_interp::{GuestMem, HostProc, Value};

/// The deterministic **scripted** backend as a [`HostCap`]: a `(argv-prefix → {stdout, stderr,
/// exit})` table, no host processes. What differential tests and wasm/browser embedders grant.
pub fn scripted_exec(table: Vec<ScriptedEntry>) -> HostCap {
    HostCap::host_proc(0, svm_exec::scripted_exec_handler(table))
}

/// The **real subprocess** backend: spawn via the host, attenuated by an explicit program
/// **allowlist** — `run`'s `argv[0]` must match an entry exactly, or the op is refused
/// (`-EPERM`). An **empty allowlist means "any program"** and is a choice the embedder must
/// spell out at the grant site. v1 is blocking one-shot: `run` feeds `stdin`, waits for exit,
/// and captures stdout/stderr; reads then drain the captured bytes. The exit code is collapsed
/// the POSIX-shell way (0-255; signal death as `128 + signo`).
pub fn host_exec(allowlist: &[&str]) -> HostCap {
    let allow: Arc<Vec<String>> = Arc::new(allowlist.iter().map(|s| s.to_string()).collect());
    HostCap::host_proc(0, host_exec_handler(allow))
}

/// One program in a [`domain_exec`] registry: the name a `run`'s `argv[0]` selects (exact
/// match — the registry *is* the attenuation, like `host_exec`'s allowlist), the instantiated
/// module to run, and the per-run [`Limits`] (a fuel bound is what makes a domain run
/// deterministic — EXEC.md).
#[derive(Clone)]
pub struct DomainProgram {
    pub name: String,
    pub instance: Arc<Instance>,
    pub limits: Limits,
}

/// The **child-domain** backend (EXEC.md's third row): a spawn is a fresh svm domain — its own
/// window, powerbox, and fuel — run to completion with the wire `stdin` seeded and both output
/// streams captured, no OS process anywhere. `argv[0]` resolves through the program registry
/// (a miss is `-EPERM`, the same refusal shape as an allowlist or table miss); the full argv
/// rides the §3e args buffer, so a `main(int, char**)` program reads it exactly as it would
/// standalone. The exit code is the child's entry result **verbatim** (`Exit(code)` or the
/// first returned value; nothing collapsed) per the contract's exit-code table; a child that
/// **traps** is a failed `run` (`-EINVAL`, probeable) — v1 does not invent an exit code for a
/// crash. Runs on the reference interpreter (deterministic under fuel; a serving module would
/// fold there anyway). Blocking profile: v1 `run` executes the child inline, exactly as
/// `host_exec` blocks on the process — one synchronous op either way.
pub fn domain_exec(programs: Vec<DomainProgram>) -> HostCap {
    domain_exec_inner(programs, None)
}

/// Like [`domain_exec`], but each spawned child additionally inherits `child_fs` — an explicitly
/// provided `fs` capability, granted to the child under the name `"fs"`. Back it with a
/// [`crate::fs::mem_fs_shared_factory`]-derived cap and every phase shares **one** working
/// filesystem, so phase N writes `x.nif` and phase N+1 reads it — the faithful file-based `nifmake`
/// hand-off (NIM.md §3c). This is a **deliberate widening of a child's authority** and lives at the
/// grant site by design: plain [`domain_exec`] gives a child only stdin/stdout, and a child never
/// gains `fs` unless the embedder hands it one here. Even then the child gets *only* this fs (the
/// in-memory, seeded store the cap wraps) — no host filesystem, no ambient authority; a child cannot
/// widen its own grant. The children share the store the caller chose to share; isolation between
/// unrelated pipelines is the caller granting them *different* factories.
pub fn domain_exec_with_fs(programs: Vec<DomainProgram>, child_fs: HostCap) -> HostCap {
    domain_exec_inner(programs, Some(child_fs))
}

fn domain_exec_inner(programs: Vec<DomainProgram>, child_fs: Option<HostCap>) -> HostCap {
    let programs = Arc::new(programs);
    let child_fs = Arc::new(child_fs);
    HostCap::host_proc(0, move || {
        let programs = Arc::clone(&programs);
        let child_fs = Arc::clone(&child_fs);
        let mut jobs = JobTable::default();
        Box::new(
            move |op: u32,
                  args: &[i64],
                  mem: Option<&mut dyn GuestMem>,
                  _minter: Option<&mut dyn svm_interp::RegionMinter>| {
                Ok(vec![domain_dispatch(
                    &programs,
                    (*child_fs).as_ref(),
                    &mut jobs,
                    op,
                    args,
                    mem,
                )])
            },
        ) as HostProc
    })
}

fn domain_dispatch(
    programs: &[DomainProgram],
    child_fs: Option<&HostCap>,
    jobs: &mut JobTable,
    op: u32,
    args: &[i64],
    mem: Option<&mut dyn GuestMem>,
) -> i64 {
    if op != EXEC_RUN {
        return jobs.handle(op, args, mem);
    }
    let (argv, stdin) = match run_args(args, mem.as_deref()) {
        Ok(x) => x,
        Err(e) => return e,
    };
    let Some(p) = programs.iter().find(|p| p.name == argv[0]) else {
        return -EPERM; // outside the registry: refused, not a trap
    };
    let config = RunConfig {
        limits: p.limits.clone(),
        stdin,
        memory_size_log2: None,
        args: argv.into_iter().map(String::into_bytes).collect(),
        env: Vec::new(),
        ..RunConfig::default()
    };
    // The child inherits only what the parent granted: stdin/stdout always, and `fs` iff the embedder
    // built this backend with `domain_exec_with_fs`. A fresh grant from the (shared) fs cap per spawn,
    // so all phases reach one store.
    let child_run = match child_fs {
        Some(fs) => p
            .instance
            .run_with_caps(Backend::TreeWalk, &config, &[("fs", fs.clone())]),
        None => p.instance.run(Backend::TreeWalk, &config),
    };
    match child_run {
        Ok(run) => {
            let exit = match run.outcome {
                Outcome::Exited(c) => c as i64,
                Outcome::Returned(vals) => vals.first().map_or(0, |v| match v {
                    Value::I64(x) => *x,
                    Value::I32(x) => *x as i64,
                    _ => 0,
                }),
            };
            jobs.push(Job {
                stdout: run.stdout,
                stderr: run.stderr,
                out_pos: 0,
                err_pos: 0,
                exit,
            })
        }
        Err(_) => -EINVAL, // the child trapped / ran out of fuel: a failed run, probeable
    }
}

fn host_exec_handler(allow: Arc<Vec<String>>) -> impl Fn() -> HostProc + Send + Sync + 'static {
    move || {
        let allow = Arc::clone(&allow);
        let mut jobs = JobTable::default();
        Box::new(
            move |op: u32,
                  args: &[i64],
                  mem: Option<&mut dyn GuestMem>,
                  _minter: Option<&mut dyn svm_interp::RegionMinter>| {
                Ok(vec![host_dispatch(&allow, &mut jobs, op, args, mem)])
            },
        ) as HostProc
    }
}

fn host_dispatch(
    allow: &[String],
    jobs: &mut JobTable,
    op: u32,
    args: &[i64],
    mem: Option<&mut dyn GuestMem>,
) -> i64 {
    if op != EXEC_RUN {
        return jobs.handle(op, args, mem);
    }
    let (argv, stdin) = match run_args(args, mem.as_deref()) {
        Ok(x) => x,
        Err(e) => return e,
    };
    if !allow.is_empty() && !allow.iter().any(|a| a == &argv[0]) {
        return -EPERM; // outside the allowlist: refused, not a trap
    }
    match spawn_blocking(&argv, &stdin) {
        Ok(job) => jobs.push(job),
        Err(e) => e,
    }
}

/// Run `argv` to completion with `stdin` fed to it; capture both output streams and collapse
/// the wait status the POSIX-shell way.
fn spawn_blocking(argv: &[String], stdin: &[u8]) -> Result<Job, i64> {
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| -ENOENT)?;
    // Feed stdin and close it so the child sees EOF. v1 payloads are one-shot and bounded by
    // the guest window, so a straight write before reaping is fine (a child that blocks writing
    // output before draining stdin could wedge only if the payload exceeds the OS pipe buffer —
    // acceptable for v1's shell-out shape; incremental stdin is the reserved extension).
    if let Some(mut si) = child.stdin.take() {
        let _ = si.write_all(stdin);
    }
    let out = child.wait_with_output().map_err(|_| -EINVAL)?;
    let exit = {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            match (out.status.code(), out.status.signal()) {
                (Some(c), _) => (c & 0xff) as i64,
                (None, Some(sig)) => 128 + sig as i64,
                (None, None) => -EINVAL,
            }
        }
        #[cfg(not(unix))]
        {
            out.status.code().map_or(-EINVAL, |c| c as i64)
        }
    };
    Ok(Job {
        stdout: out.stdout,
        stderr: out.stderr,
        out_pos: 0,
        err_pos: 0,
        exit,
    })
}
