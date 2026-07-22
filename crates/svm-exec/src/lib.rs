//! **Deterministic `exec` capability backend + the shared exec-cap wire protocol.**
//!
//! This crate holds the wasm-safe half of the subprocess capability (EXEC.md): the op-code /
//! argv / errno protocol every backend speaks, and the deterministic **scripted** backend
//! (`scripted_exec`) — a `(argv-prefix → {stdout, stderr, exit})` table, no processes at all.
//! It depends only on `svm-interp` (`HostFn`/`GuestMem`), so it builds for **wasm** — the
//! browser cdylib grants a `scripted_exec` directly. `svm-run` keeps the real subprocess
//! `host_exec(allowlist)` backend (`crates/svm-run/src/exec.rs`) and wraps these handlers in
//! its `HostCap`; it re-exports this crate so `svm_run::exec::*` is one surface.
//!
//! One interface, wirer-chosen backend (EXEC.md): a guest resolves `"exec"` by name and cannot
//! tell which backend serves it — the interposition-invisibility property the capability model
//! guarantees. v1 is **blocking one-shot**: `run` executes to completion with the given stdin;
//! `read_out`/`read_err` drain the captured output in chunks; `status` reports the exit code.
//! The contract *reserves* streaming (a future backend may yield output before exit) — guests
//! must not depend on output being complete before `status` returns 0-EOF reads.
//!
//! A handler builder returns a `make: impl Fn() -> HostFn` closure — `svm-run` passes it to
//! `HostCap::host_fn`, and the browser cdylib grants the `HostFn` directly on its Host.

use std::sync::Arc;
use svm_interp::{GuestMem, HostFn};

/// `run(argv_ptr, argv_len, stdin_ptr, stdin_len) -> job | -errno` — spawn with `argv`
/// (NUL-separated bytes, `argv[0]` the program) and `stdin` fed to it; v1 runs the job to
/// completion before returning. A program the capability does not permit (allowlist miss /
/// no table entry) is `-EPERM` — a refused op, never a trap.
pub const EXEC_RUN: u32 = 0;
/// `read_out(job, buf_ptr, buf_cap) -> n | 0 = EOF | -errno` — copy the next chunk of the
/// job's stdout into the guest buffer.
pub const EXEC_READ_OUT: u32 = 1;
/// `read_err(job, buf_ptr, buf_cap) -> n | 0 = EOF | -errno` — same for stderr.
pub const EXEC_READ_ERR: u32 = 2;
/// `status(job) -> exit code | -errno` — the job's exit code (POSIX-shell collapse on the
/// host backend: 0-255, signal death as 128+signo; verbatim on scripted/domain backends).
pub const EXEC_STATUS: u32 = 3;
/// `close(job) -> 0 | -errno` — release the job handle.
pub const EXEC_CLOSE: u32 = 4;

pub const EPERM: i64 = 1;
pub const ENOENT: i64 = 2;
pub const EBADF: i64 = 9;
pub const EFAULT: i64 = 14;
pub const EINVAL: i64 = 22;

/// Parse the wire `argv`: NUL-separated byte strings, no trailing NUL required, each a
/// non-empty UTF-8 token. `argv[0]` is the program. Empty argv or a non-UTF-8 token is
/// `-EINVAL`; an unreadable buffer is `-EFAULT`.
pub fn read_argv(mem: Option<&dyn GuestMem>, ptr: i64, len: i64) -> Result<Vec<String>, i64> {
    let mem = mem.ok_or(-EFAULT)?;
    if !(1..=65536).contains(&len) || ptr < 0 {
        return Err(-EINVAL);
    }
    let bytes = mem.read_bytes(ptr as u64, len as u64).ok_or(-EFAULT)?;
    let argv: Vec<String> = bytes
        .split(|&b| b == 0)
        .filter(|t| !t.is_empty())
        .map(|t| String::from_utf8(t.to_vec()).map_err(|_| -EINVAL))
        .collect::<Result<_, i64>>()?;
    if argv.is_empty() {
        return Err(-EINVAL);
    }
    Ok(argv)
}

/// Read a guest byte buffer (`stdin` payload); `len == 0` is an empty stdin, not an error.
pub fn read_bytes(mem: Option<&dyn GuestMem>, ptr: i64, len: i64) -> Result<Vec<u8>, i64> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let mem = mem.ok_or(-EFAULT)?;
    if len < 0 || ptr < 0 {
        return Err(-EINVAL);
    }
    mem.read_bytes(ptr as u64, len as u64).ok_or(-EFAULT)
}

/// One completed job: captured output with read cursors + the exit code. v1 jobs are complete
/// at creation (`run` is synchronous-to-completion), so reads only ever drain buffers.
pub struct Job {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub out_pos: usize,
    pub err_pos: usize,
    pub exit: i64,
}

/// Shared per-grant job table + op dispatch: everything except *how a job is produced*.
/// Both backends (`scripted_exec` here, `host_exec` in `svm-run`) route their non-`run` ops
/// through this, so read/status/close semantics are backend-identical by construction.
#[derive(Default)]
pub struct JobTable {
    pub jobs: Vec<Option<Job>>,
}

impl JobTable {
    /// Install a completed job, returning its handle.
    pub fn push(&mut self, job: Job) -> i64 {
        match self.jobs.iter().position(Option::is_none) {
            Some(i) => {
                self.jobs[i] = Some(job);
                i as i64
            }
            None => {
                self.jobs.push(Some(job));
                (self.jobs.len() - 1) as i64
            }
        }
    }

    /// Dispatch a non-`run` op (`read_out`/`read_err`/`status`/`close`).
    pub fn handle(&mut self, op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>) -> i64 {
        let job_idx = match args.first() {
            Some(&j) if j >= 0 && (j as usize) < self.jobs.len() => j as usize,
            _ => return -EBADF,
        };
        let Some(job) = self.jobs[job_idx].as_mut() else {
            return -EBADF;
        };
        match op {
            EXEC_READ_OUT | EXEC_READ_ERR => {
                let (&ptr, &cap) = match (args.get(1), args.get(2)) {
                    (Some(p), Some(c)) if *p >= 0 && *c >= 0 => (p, c),
                    _ => return -EINVAL,
                };
                let (buf, pos) = if op == EXEC_READ_OUT {
                    (&job.stdout, &mut job.out_pos)
                } else {
                    (&job.stderr, &mut job.err_pos)
                };
                let n = (buf.len() - *pos).min(cap as usize);
                if n == 0 {
                    return 0; // EOF
                }
                let Some(mem) = mem else { return -EFAULT };
                if mem.write_bytes(ptr as u64, &buf[*pos..*pos + n]).is_none() {
                    return -EFAULT;
                }
                *pos += n;
                n as i64
            }
            EXEC_STATUS => job.exit,
            EXEC_CLOSE => {
                self.jobs[job_idx] = None;
                0
            }
            _ => -EINVAL,
        }
    }
}

/// One scripted entry: an argv **prefix** and the result any matching `run` yields.
#[derive(Clone)]
pub struct ScriptedEntry {
    pub argv_prefix: Vec<String>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit: i64,
}

/// A `make` builder for the deterministic **scripted** backend (EXEC.md): `run` matches the
/// wire argv against the table (longest matching argv-prefix wins) and yields that entry's
/// `{stdout, stderr, exit}` — no host process is ever created. A miss is `-EPERM` (the same
/// refused-op shape as a `host_exec` allowlist miss, so a guest cannot distinguish backends
/// by their failure mode). Each grant gets a fresh job table; the table itself is shared and
/// immutable, so re-runs are deterministic.
pub fn scripted_exec_handler(
    table: Vec<ScriptedEntry>,
) -> impl Fn() -> HostFn + Send + Sync + 'static {
    let table = Arc::new(table);
    move || {
        let table = Arc::clone(&table);
        let mut jobs = JobTable::default();
        Box::new(
            move |op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>| {
                Ok(vec![scripted_dispatch(&table, &mut jobs, op, args, mem)])
            },
        ) as HostFn
    }
}

fn scripted_dispatch(
    table: &[ScriptedEntry],
    jobs: &mut JobTable,
    op: u32,
    args: &[i64],
    mem: Option<&mut dyn GuestMem>,
) -> i64 {
    if op != EXEC_RUN {
        return jobs.handle(op, args, mem);
    }
    match run_args(args, mem.as_deref()) {
        Ok((argv, _stdin)) => {
            let hit = table
                .iter()
                .filter(|e| argv.starts_with(&e.argv_prefix))
                .max_by_key(|e| e.argv_prefix.len());
            match hit {
                Some(e) => jobs.push(Job {
                    stdout: e.stdout.clone(),
                    stderr: e.stderr.clone(),
                    out_pos: 0,
                    err_pos: 0,
                    exit: e.exit,
                }),
                None => -EPERM,
            }
        }
        Err(e) => e,
    }
}

/// Decode `run`'s `(argv_ptr, argv_len, stdin_ptr, stdin_len)` into `(argv, stdin_bytes)` —
/// shared by both backends so the wire shape cannot drift.
pub fn run_args(args: &[i64], mem: Option<&dyn GuestMem>) -> Result<(Vec<String>, Vec<u8>), i64> {
    let (&ap, &al, &sp, &sl) = match (args.first(), args.get(1), args.get(2), args.get(3)) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => return Err(-EINVAL),
    };
    let argv = read_argv(mem, ap, al)?;
    let stdin = read_bytes(mem, sp, sl)?;
    Ok((argv, stdin))
}
