//! A **POSIX personality** as an embedder host capability (POSIX.md, §7).
//!
//! The "host provides libc as a capability; the §7 named-import mechanism carries it" story — the
//! same shape as [`svm-wasi`](../svm_wasi), generalized from a WASI subset to the POSIX/libc surface a
//! fork-less shell (BusyBox `ash` → Bash) links against. [`resolve`] binds libc symbol names to a
//! single [`svm_interp::iface::HOST_FN`] capability; [`handler`] implements the ops over the guest
//! window. All libc *semantics* live **here** — outside the interp escape-TCB — reached only through a
//! granted, masked, type-checked handle (DESIGN.md §7).
//!
//! **State model (POSIX.md §3):** the bytes a libc call touches — a `malloc`'d buffer, a `write`
//! source — live in the **guest window** (native-speed access; `malloc` returns a window offset). The
//! *bookkeeping* — the allocator's cursor, captured stdout/stderr, the stdin cursor — lives host-side
//! in [`Inner`], never in the guest's address space, so the guest cannot corrupt it.
//!
//! Scope: `write` / `read` / `malloc` / `free` / `exit`, plus `open` / `close` / `lseek` / `unlink`
//! over an in-memory filesystem (a `path → bytes` memfs) with a host-side fd table, and
//! `getcwd` / `chdir` / `getenv` / `setenv` over a host-side cwd + environment. `malloc` is a
//! first-fit free list over a configured window-heap region. Still to come (POSIX.md §6):
//! `stat`/`readdir`, signals, and `fork`/`exec`. Pure computation (`strlen`, `snprintf`, `math`, …)
//! is **guest code**, not a cap — it needs no authority (POSIX.md §1).
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use svm_interp::{iface, GuestMem, Host, HostFn, Trap};
use svm_ir::ResolvedCap;

/// Op numbers on the shared `HOST_FN` handle; [`resolve`] maps libc names to these.
pub const OP_WRITE: u32 = 0;
pub const OP_READ: u32 = 1;
pub const OP_MALLOC: u32 = 2;
pub const OP_FREE: u32 = 3;
pub const OP_EXIT: u32 = 4;
pub const OP_OPEN: u32 = 5;
pub const OP_CLOSE: u32 = 6;
pub const OP_LSEEK: u32 = 7;
pub const OP_UNLINK: u32 = 8;
pub const OP_GETCWD: u32 = 9;
pub const OP_CHDIR: u32 = 10;
pub const OP_GETENV: u32 = 11;
pub const OP_SETENV: u32 = 12;
pub const OP_STAT: u32 = 13;
pub const OP_OPENDIR: u32 = 14;
pub const OP_READDIR: u32 = 15;
pub const OP_CLOSEDIR: u32 = 16;
pub const OP_ARGC: u32 = 17;
pub const OP_ARGV: u32 = 18;

/// Negative errnos this personality returns (Linux values, so a guest's `<errno.h>` agrees).
const ENOENT: i64 = -2; // no such file (open without O_CREAT; stat/opendir of an absent path)
const EBADF: i64 = -9; // an op on an fd this personality does not serve
const EINVAL: i64 = -22; // bad argument (whence, non-UTF-8 path, negative seek)
const ENOTDIR: i64 = -20; // opendir on a path that is a regular file, not a directory
const ERANGE: i64 = -34; // result won't fit the caller's buffer (getcwd)

/// `struct stat` **mode** bits this personality reports (Linux `<sys/stat.h>` `S_IFMT` values). The
/// personality's `struct stat` is a deliberately minimal **`{ i64 st_mode; i64 st_size; }`** (16
/// bytes) — the two fields a shell actually reads (`S_ISDIR`/`S_ISREG` on `st_mode`, `st_size`); a
/// guest `<sys/stat.h>` agrees on that layout (POSIX.md §5). A memfs file is a regular file; a path
/// that is a prefix of some file key (or `"/"`) is a directory.
const S_IFREG: i64 = 0o100000; // regular file (| 0o644 perms)
const S_IFDIR: i64 = 0o040000; // directory (| 0o755 perms)

// The ABI is **explicit-length**, syscall-style: a string argument is `(ptr, len)`, not a
// NUL-terminated `char*`. This avoids an unbounded window scan (safer) and matches `read`/`write`;
// a thin guest libc adapts C's NUL-terminated conventions to it (POSIX.md §4, "one ABI, two
// bindings" — the shim is guest code). `getcwd`/`getenv` *write* NUL-terminated results (C's
// contract) since the caller consumes them as `char*`.

/// `open` flags (Linux `<fcntl.h>` values). The low two bits are the access mode.
const O_ACCMODE: i64 = 3;
const O_WRONLY: i64 = 1;
const O_RDWR: i64 = 2;
const O_CREAT: i64 = 0o100;
const O_TRUNC: i64 = 0o1000;
const O_APPEND: i64 = 0o2000;

/// `lseek` whence values (`SEEK_SET`/`SEEK_CUR`/`SEEK_END`).
const SEEK_SET: i64 = 0;
const SEEK_CUR: i64 = 1;
const SEEK_END: i64 = 2;

/// The first fd the file table hands out — `0`/`1`/`2` are the reserved stdio streams.
const FIRST_FD: usize = 3;

/// One entry in the host-side fd table: which memfs file it refers to, the current offset, and whether
/// it was opened for writing. Independent offsets per fd, shared file contents (POSIX file semantics).
struct OpenFile {
    path: String,
    pos: usize,
    writable: bool,
}

/// One open directory stream: the immediate child names under the opened path, snapshotted at
/// `opendir` (so a concurrent `open`/`unlink` during iteration doesn't perturb it — POSIX permits
/// either), plus the `readdir` cursor.
struct DirStream {
    entries: Vec<String>,
    pos: usize,
}

/// The allocator's alignment (bytes). 16 covers `max_align_t` (doubles / SIMD) so a `malloc`'d buffer
/// is suitably aligned for anything the guest stores into it.
const ALIGN: u64 = 16;

/// Host-side bookkeeping for one POSIX personality: captured output, the stdin cursor, and the
/// window-heap allocator cursor. Lives outside the guest window (POSIX.md §3), shared (`Arc<Mutex>`)
/// so an embedder/test can read the captured output back after a run.
struct Inner {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    /// Preloaded standard input; `read(0, …)` drains it from `stdin_pos`.
    stdin: Vec<u8>,
    stdin_pos: usize,
    /// High-water mark: the window offset fresh (never-freed) allocations bump upward from.
    heap_next: u64,
    /// One past the last window byte the allocator may hand out.
    heap_end: u64,
    /// Live allocations, `ptr → size` — so `free` knows a block's length (the size header lives
    /// host-side, out of the guest's reach, rather than in a window prefix the guest could clobber).
    allocated: HashMap<u64, u64>,
    /// Freed blocks available for reuse (`offset, size`), first-fit. No coalescing yet — adjacent
    /// frees stay separate (a fragmentation follow-up, POSIX.md §6); reuse of a same-or-larger block
    /// works regardless.
    free_list: Vec<(u64, u64)>,
    /// The **in-memory filesystem**: path → contents. A memfs keeps the personality self-contained and
    /// deterministic (the playground has no disk); a native embedder routing to a real `fs` cap is a
    /// follow-up. Shared file bytes; per-fd offsets live in [`Inner::fds`].
    files: HashMap<String, Vec<u8>>,
    /// The host-side fd table (indexed by fd; `0`/`1`/`2` are always `None` — stdio is handled
    /// specially). `open` allocates the first free slot at [`FIRST_FD`] or above.
    fds: Vec<Option<OpenFile>>,
    /// Open directory streams (`opendir`/`readdir`/`closedir`), indexed by the `DIR*`-analog handle
    /// `opendir` returns. Each holds the immediate child names snapshotted at `opendir` time and a
    /// read cursor. Separate from [`Inner::fds`] (a directory stream is not a file fd here).
    dirs: Vec<Option<DirStream>>,
    /// The program's argument vector (`args[0]` is the program name), delivered **host-side** — the
    /// symmetric analogue of the environment: `argc`/`argv` read it, the embedder sets it. This is how
    /// a personality program gets `sh -c "…"` without the window args buffer (POSIX.md §5); a guest
    /// crt that wants a standard `main(int, char**)` builds `argv[]` from these ops.
    args: Vec<String>,
    /// The current working directory `getcwd` reports and `chdir` updates. A plain string — the memfs
    /// is flat (paths are used as-given), so `cwd` is not validated against it; path normalization/
    /// resolution is a follow-up (POSIX.md §6).
    cwd: String,
    /// The environment: `name → value`. `getenv`/`setenv` read and update it; host-side, out of the
    /// guest's reach, like the rest of the bookkeeping (POSIX.md §3).
    env: HashMap<String, String>,
    /// Cache of `getenv` results already materialized into the window: `name → ptr`. C's `getenv`
    /// returns a stable `char*` into libc-owned storage, so a repeated `getenv("X")` must return the
    /// **same** pointer; we allocate a NUL-terminated copy in the arena once and reuse it. `setenv`
    /// invalidates the entry so the next `getenv` re-materializes the new value.
    env_ptrs: HashMap<String, u64>,
}

/// A handle to a granted POSIX personality's shared state — read the captured output after a run.
/// Cheap to clone (shares one `Arc`).
#[derive(Clone)]
pub struct Posix {
    inner: Arc<Mutex<Inner>>,
}

impl Posix {
    /// Bytes the guest `write`-to-fd-1'd.
    pub fn stdout(&self) -> Vec<u8> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stdout
            .clone()
    }
    /// Bytes the guest `write`-to-fd-2'd.
    pub fn stderr(&self) -> Vec<u8> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stderr
            .clone()
    }

    /// Seed (or overwrite) a memfs file — how an embedder/test stages the filesystem a guest `open`s.
    pub fn write_file(&self, path: &str, bytes: &[u8]) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .files
            .insert(path.to_string(), bytes.to_vec());
    }

    /// Read a memfs file back — how an embedder/test inspects what the guest wrote.
    pub fn read_file(&self, path: &str) -> Option<Vec<u8>> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .files
            .get(path)
            .cloned()
    }

    /// Seed (or overwrite) an environment variable — how an embedder/test stages the environment a
    /// guest `getenv`s. Invalidates any cached `getenv` pointer for the name.
    pub fn set_env(&self, name: &str, value: &str) {
        let mut st = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        st.env_ptrs.remove(name);
        st.env.insert(name.to_string(), value.to_string());
    }

    /// The current working directory — how an embedder/test observes a guest `chdir`.
    pub fn cwd(&self) -> String {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cwd
            .clone()
    }

    /// Set the program's argument vector (`args[0]` is conventionally the program name) — how an
    /// embedder hands a personality program its `argv` (e.g. `["sh", "-c", "echo hi"]`), read back by
    /// the guest through the `argc`/`argv` ops.
    pub fn set_args(&self, args: &[&str]) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).args =
            args.iter().map(|s| s.to_string()).collect();
    }
}

/// The §7 import-name resolver for the POSIX subset: binds libc symbol names to the
/// [`iface::HOST_FN`] capability + op. Pass it to [`svm_ir::resolve_imports`] (via
/// `svm_run::resolve_capability_imports`); compose with your own policy for other imports. Unknown
/// names return `None`, so `resolve_imports` fails closed. Both bare (`"write"`) and `"posix."`-
/// prefixed names resolve, so it works whether the frontend emits raw libc symbols or namespaced ones.
pub fn resolve(name: &str) -> Option<ResolvedCap> {
    let bare = name.strip_prefix("posix.").unwrap_or(name);
    let op = match bare {
        "write" => OP_WRITE,
        "read" => OP_READ,
        "malloc" => OP_MALLOC,
        "free" => OP_FREE,
        "exit" | "_exit" | "_Exit" => OP_EXIT,
        "open" => OP_OPEN,
        "close" => OP_CLOSE,
        "lseek" => OP_LSEEK,
        "unlink" | "remove" => OP_UNLINK,
        "getcwd" => OP_GETCWD,
        "chdir" => OP_CHDIR,
        "getenv" => OP_GETENV,
        "setenv" => OP_SETENV,
        "stat" | "lstat" => OP_STAT,
        "opendir" => OP_OPENDIR,
        "readdir" => OP_READDIR,
        "closedir" => OP_CLOSEDIR,
        // Personality extensions (not standard libc functions): the host-side argument vector, the
        // symmetric analogue of `getenv`/`environ`. A guest crt reads these to build `main`'s argv.
        "argc" => OP_ARGC,
        "argv" => OP_ARGV,
        _ => return None,
    };
    Some(ResolvedCap {
        type_id: iface::HOST_FN,
        op,
    })
}

/// The §7 **general-form** resolver (DESIGN.md §7 late binding: a name resolves to "a registered
/// implementation **+ handle**"): like [`resolve`], but also binds the granted personality `handle`,
/// so a module's libc imports carry **no handle argument** — each `call.import`'s handle operand is a
/// `ConstI32` placeholder patched at resolve ([`svm_ir::Resolved::CapBound`]). Pass the closure to
/// [`svm_ir::resolve_imports_with`] **after** [`grant`] (resolution needs the granted handle — the §7
/// "binding happens once, at instantiation" ordering). The guest declares plain libc signatures and
/// never reads a powerbox slot; the import section is its capability manifest.
pub fn resolve_bound(handle: i32) -> impl Fn(&str) -> Option<svm_ir::Resolved> {
    move |name| {
        resolve(name).map(|c| svm_ir::Resolved::CapBound {
            type_id: c.type_id,
            op: c.op,
            handle,
        })
    }
}

/// Grant a POSIX personality on `host`, returning the `HOST_FN` handle and a [`Posix`] handle to its
/// captured state. `heap_base`/`heap_end` bound the window region `malloc` hands out (both window
/// offsets, `heap_base <= heap_end`, within the guest window and clear of the guest's static
/// data/stack). `stdin` preloads standard input for `read(0, …)`. Every libc import in a linked module
/// shares this **one** handle (svm-wasm/chibicc thread a single capability handle); the op number
/// distinguishes the call, so pass the handle as the entry's leading argument.
pub fn grant(host: &mut Host, heap_base: u64, heap_end: u64, stdin: Vec<u8>) -> (i32, Posix) {
    let inner = Arc::new(Mutex::new(Inner {
        stdout: Vec::new(),
        stderr: Vec::new(),
        stdin,
        stdin_pos: 0,
        heap_next: heap_base,
        heap_end,
        allocated: HashMap::new(),
        free_list: Vec::new(),
        files: HashMap::new(),
        fds: Vec::new(),
        dirs: Vec::new(),
        args: Vec::new(),
        cwd: "/".to_string(),
        env: HashMap::new(),
        env_ptrs: HashMap::new(),
    }));
    let posix = Posix {
        inner: Arc::clone(&inner),
    };
    let handle = host.grant_host_fn(handler(inner));
    (handle, posix)
}

/// Build the POSIX [`HostFn`] handler over shared `inner`. Dispatches on the op number; an unknown op
/// on this handle is a `CapFault` (as for any capability).
fn handler(inner: Arc<Mutex<Inner>>) -> HostFn {
    Box::new(move |op, args, mem| {
        let mut st = inner.lock().unwrap_or_else(|e| e.into_inner());
        match op {
            OP_WRITE => st.write(args, mem),
            OP_READ => st.read(args, mem),
            OP_MALLOC => Ok(vec![st.malloc(args)]),
            OP_FREE => {
                st.free(args);
                Ok(vec![0])
            }
            OP_EXIT => Err(Trap::Exit(args.first().copied().unwrap_or(0) as i32)),
            OP_OPEN => st.open(args, mem),
            OP_CLOSE => Ok(vec![st.close(args)]),
            OP_LSEEK => Ok(vec![st.lseek(args)]),
            OP_UNLINK => st.unlink(args, mem),
            OP_STAT => st.stat(args, mem),
            OP_OPENDIR => st.opendir(args, mem),
            OP_READDIR => st.readdir(args, mem),
            OP_CLOSEDIR => Ok(vec![st.closedir(args)]),
            OP_ARGC => Ok(vec![st.args.len() as i64]),
            OP_ARGV => st.argv(args, mem),
            OP_GETCWD => st.getcwd(args, mem),
            OP_CHDIR => st.chdir(args, mem),
            OP_GETENV => st.getenv(args, mem),
            OP_SETENV => st.setenv(args, mem),
            _ => Err(Trap::CapFault),
        }
    })
}

impl Inner {
    /// `write(fd, buf, len) -> n | -errno`: `1`/`2` append to the captured stdout/stderr; an fd `>= 3`
    /// writes into its memfs file at the fd's offset (extending it), advancing the offset. `0` (stdin)
    /// and an unopened / read-only fd are `-EBADF`.
    fn write(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let fd = *args.first().ok_or(Trap::Malformed)?;
        let buf = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let len = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as u64;
        if len == 0 {
            return Ok(vec![0]);
        }
        let data = mem.read_bytes(buf, len).ok_or(Trap::Malformed)?;
        match fd {
            1 => self.stdout.extend_from_slice(&data),
            2 => self.stderr.extend_from_slice(&data),
            f if f >= FIRST_FD as i64 => return Ok(vec![self.file_write(f as usize, &data)]),
            _ => return Ok(vec![EBADF]),
        }
        Ok(vec![len as i64])
    }

    /// `read(fd, buf, len) -> n | -errno`: `0` drains preloaded stdin; an fd `>= 3` reads its memfs file
    /// from the fd's offset, advancing it (`0` at EOF). `1`/`2` and an unopened fd are `-EBADF`.
    fn read(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let fd = *args.first().ok_or(Trap::Malformed)?;
        let buf = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let len = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as usize;
        let chunk: Vec<u8> = match fd {
            0 => {
                let avail = &self.stdin[self.stdin_pos.min(self.stdin.len())..];
                let n = len.min(avail.len());
                self.stdin_pos += n;
                avail[..n].to_vec()
            }
            f if f >= FIRST_FD as i64 => match self.file_read(f as usize, len) {
                Ok(c) => c,
                Err(e) => return Ok(vec![e]),
            },
            _ => return Ok(vec![EBADF]),
        };
        mem.write_bytes(buf, &chunk).ok_or(Trap::Malformed)?;
        Ok(vec![chunk.len() as i64])
    }

    /// `open(path_ptr, path_len, flags) -> fd | -errno`: open (or `O_CREAT`) a memfs file, returning a
    /// fresh fd. `O_TRUNC` clears it, `O_APPEND` seeks to the end; a missing file without `O_CREAT` is
    /// `-ENOENT`, a non-UTF-8 path `-EINVAL`.
    fn open(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let flags = *args.get(2).ok_or(Trap::Malformed)?;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        let exists = self.files.contains_key(&path);
        if !exists && flags & O_CREAT == 0 {
            return Ok(vec![ENOENT]);
        }
        let file = self.files.entry(path.clone()).or_default();
        if flags & O_TRUNC != 0 {
            file.clear();
        }
        let pos = if flags & O_APPEND != 0 { file.len() } else { 0 };
        let acc = flags & O_ACCMODE;
        let writable = acc == O_WRONLY || acc == O_RDWR;
        Ok(vec![self.alloc_fd(OpenFile {
            path,
            pos,
            writable,
        })])
    }

    /// `close(fd) -> 0 | -errno`: release a file fd. stdio / unopened fds are `-EBADF`.
    fn close(&mut self, args: &[i64]) -> i64 {
        let fd = *args.first().unwrap_or(&-1);
        if fd >= FIRST_FD as i64 {
            if let Some(slot @ Some(_)) = self.fds.get_mut(fd as usize) {
                *slot = None;
                return 0;
            }
        }
        EBADF
    }

    /// `lseek(fd, offset, whence) -> new_offset | -errno`: reposition a file fd (`SEEK_SET`/`CUR`/`END`).
    /// A negative result or bad whence is `-EINVAL`; stdio / unopened fds are `-EBADF`.
    fn lseek(&mut self, args: &[i64]) -> i64 {
        let fd = *args.first().unwrap_or(&-1);
        let offset = *args.get(1).unwrap_or(&0);
        let whence = *args.get(2).unwrap_or(&-1);
        if fd < FIRST_FD as i64 {
            return EBADF;
        }
        let (path, pos) = match self.fds.get(fd as usize).and_then(|s| s.as_ref()) {
            Some(of) => (of.path.clone(), of.pos as i64),
            None => return EBADF,
        };
        let size = self.files.get(&path).map_or(0, |f| f.len()) as i64;
        let newpos = match whence {
            SEEK_SET => offset,
            SEEK_CUR => pos + offset,
            SEEK_END => size + offset,
            _ => return EINVAL,
        };
        if newpos < 0 {
            return EINVAL;
        }
        self.fds[fd as usize].as_mut().unwrap().pos = newpos as usize;
        newpos
    }

    /// `unlink(path_ptr, path_len) -> 0 | -errno`: remove a memfs file. Already-open fds keep their
    /// (now-detached) contents via the file map only until closed — POSIX unlink-while-open nuance is a
    /// follow-up; here a removed path simply reads as absent to a fresh `open`. Missing file is `-ENOENT`.
    fn unlink(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        Ok(vec![if self.files.remove(&path).is_some() {
            0
        } else {
            ENOENT
        }])
    }

    /// The immediate child **names** of directory `path` in the flat memfs — the distinct first
    /// component of every file key under `path` (deduped, sorted for determinism). A file key exactly
    /// one level below yields its basename; a key deeper below yields the intervening subdir name
    /// (so a directory appears once even with many files under it). `"/"` lists top-level components.
    fn dir_children(&self, path: &str) -> Vec<String> {
        // Normalize to the prefix every child key starts with: `path` + "/" (just "/" for the root).
        let prefix = if path == "/" {
            "/".to_string()
        } else {
            format!("{}/", path.trim_end_matches('/'))
        };
        let mut names: Vec<String> = self
            .files
            .keys()
            .filter_map(|k| k.strip_prefix(&prefix))
            .filter(|rest| !rest.is_empty())
            .map(|rest| rest.split('/').next().unwrap_or(rest).to_string())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// True if `path` names a directory in the flat memfs: the root `"/"`, or any path that is a
    /// proper prefix of some file key (i.e. has at least one child). Not a file key itself.
    fn is_dir(&self, path: &str) -> bool {
        path == "/" || !self.dir_children(path).is_empty()
    }

    /// `stat(path_ptr, path_len, statbuf_ptr) -> 0 | -errno`: fill the caller's `struct stat`
    /// (`{ i64 st_mode; i64 st_size; }`, 16 bytes) for a memfs path. A file key is `S_IFREG` with its
    /// byte length; a directory (a prefix of some key, or `"/"`) is `S_IFDIR` size 0; anything else is
    /// `-ENOENT`. A non-UTF-8 path is `-EINVAL`.
    fn stat(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let buf = *args.get(2).ok_or(Trap::Malformed)? as u64;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        let (mode, size) = if let Some(f) = self.files.get(&path) {
            (S_IFREG | 0o644, f.len() as i64)
        } else if self.is_dir(&path) {
            (S_IFDIR | 0o755, 0)
        } else {
            return Ok(vec![ENOENT]);
        };
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&mode.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        mem.write_bytes(buf, &out).ok_or(Trap::Malformed)?;
        Ok(vec![0])
    }

    /// `opendir(path_ptr, path_len) -> dir | -errno`: snapshot a directory's immediate children and
    /// return a `DIR*`-analog handle for `readdir`/`closedir`. A regular file is `-ENOTDIR`; a path
    /// with no children that isn't the root is `-ENOENT`; a non-UTF-8 path is `-EINVAL`.
    fn opendir(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        if self.files.contains_key(&path) {
            return Ok(vec![ENOTDIR]);
        }
        if !self.is_dir(&path) {
            return Ok(vec![ENOENT]);
        }
        let entries = self.dir_children(&path);
        let stream = DirStream { entries, pos: 0 };
        let idx = match self.dirs.iter().position(Option::is_none) {
            Some(i) => {
                self.dirs[i] = Some(stream);
                i
            }
            None => {
                self.dirs.push(Some(stream));
                self.dirs.len() - 1
            }
        };
        Ok(vec![idx as i64])
    }

    /// `readdir(dir, name_ptr, name_cap) -> namelen | 0 | -errno`: write the next entry's name
    /// (NUL-terminated, C's `dirent.d_name` convention) into the caller's buffer and advance. Returns
    /// the name length (excluding the NUL) on success, `0` at end of stream, `-EBADF` for a stale
    /// handle, `-ERANGE` if the name + NUL won't fit `name_cap`.
    fn readdir(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let dir = *args.first().ok_or(Trap::Malformed)?;
        let name_ptr = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let cap = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as u64;
        let Some(stream) = usize::try_from(dir)
            .ok()
            .and_then(|i| self.dirs.get_mut(i)?.as_mut())
        else {
            return Ok(vec![EBADF]);
        };
        let Some(name) = stream.entries.get(stream.pos) else {
            return Ok(vec![0]); // end of stream
        };
        let mut bytes = name.clone().into_bytes();
        let namelen = bytes.len() as i64;
        bytes.push(0); // NUL
        if bytes.len() as u64 > cap {
            return Ok(vec![ERANGE]);
        }
        stream.pos += 1;
        mem.write_bytes(name_ptr, &bytes).ok_or(Trap::Malformed)?;
        Ok(vec![namelen])
    }

    /// `closedir(dir) -> 0 | -errno`: release a directory stream. A stale handle is `-EBADF`.
    fn closedir(&mut self, args: &[i64]) -> i64 {
        let dir = *args.first().unwrap_or(&-1);
        if let Some(slot @ Some(_)) = usize::try_from(dir).ok().and_then(|i| self.dirs.get_mut(i)) {
            *slot = None;
            0
        } else {
            EBADF
        }
    }

    /// `argv(i, buf, cap) -> len | -errno`: write argument `i` (NUL-terminated) into the caller's
    /// buffer and return its length (excluding the NUL). An out-of-range index is `-EINVAL`; a name
    /// that won't fit `cap` is `-ERANGE`. (`argc` is a fieldless op: `self.args.len()`.)
    fn argv(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let i = *args.first().ok_or(Trap::Malformed)?;
        let buf = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let cap = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as u64;
        let Some(arg) = usize::try_from(i).ok().and_then(|i| self.args.get(i)) else {
            return Ok(vec![EINVAL]);
        };
        let mut bytes = arg.clone().into_bytes();
        let len = bytes.len() as i64;
        bytes.push(0);
        if bytes.len() as u64 > cap {
            return Ok(vec![ERANGE]);
        }
        mem.write_bytes(buf, &bytes).ok_or(Trap::Malformed)?;
        Ok(vec![len])
    }

    /// Allocate the first free fd at [`FIRST_FD`] or above for `of`, extending the table if needed.
    fn alloc_fd(&mut self, of: OpenFile) -> i64 {
        while self.fds.len() < FIRST_FD {
            self.fds.push(None);
        }
        match (FIRST_FD..self.fds.len()).find(|&i| self.fds[i].is_none()) {
            Some(i) => {
                self.fds[i] = Some(of);
                i as i64
            }
            None => {
                self.fds.push(Some(of));
                (self.fds.len() - 1) as i64
            }
        }
    }

    /// Write `data` into fd `fd`'s memfs file at its offset (extending with zeros if the offset is
    /// past the end), advancing the offset. Returns the count, or `-EBADF` for an unopened / read-only fd.
    fn file_write(&mut self, fd: usize, data: &[u8]) -> i64 {
        let (path, pos) = match self.fds.get(fd).and_then(|s| s.as_ref()) {
            Some(of) if of.writable => (of.path.clone(), of.pos),
            _ => return EBADF,
        };
        let file = self.files.entry(path).or_default();
        let end = pos + data.len();
        if file.len() < end {
            file.resize(end, 0);
        }
        file[pos..end].copy_from_slice(data);
        self.fds[fd].as_mut().unwrap().pos = end;
        data.len() as i64
    }

    /// Read up to `len` bytes from fd `fd`'s memfs file at its offset, advancing it. `Err(-errno)` for
    /// an unopened fd.
    fn file_read(&mut self, fd: usize, len: usize) -> Result<Vec<u8>, i64> {
        let (path, pos) = match self.fds.get(fd).and_then(|s| s.as_ref()) {
            Some(of) => (of.path.clone(), of.pos),
            None => return Err(EBADF),
        };
        let file = self.files.get(&path).map(|v| v.as_slice()).unwrap_or(&[]);
        let n = len.min(file.len().saturating_sub(pos));
        let chunk = file[pos..pos + n].to_vec();
        self.fds[fd].as_mut().unwrap().pos = pos + n;
        Ok(chunk)
    }

    /// `malloc(size) -> ptr | 0`: an `ALIGN`-aligned window offset from the heap arena. First-fit
    /// **reuse** of a freed block (split if larger), else **bump** from the high-water mark. `0` (the
    /// C `NULL`) when neither can satisfy the request within `heap_end` — the anti-bomb bound.
    fn malloc(&mut self, args: &[i64]) -> i64 {
        // Round the request up to `ALIGN`; a zero-size request still yields a unique non-null cell.
        let want = ((*args.first().unwrap_or(&0)).max(0) as u64)
            .max(1)
            .div_ceil(ALIGN)
            * ALIGN;
        // First-fit over the free list: reuse the first block that fits, splitting off any remainder.
        if let Some(i) = self.free_list.iter().position(|&(_, sz)| sz >= want) {
            let (off, sz) = self.free_list.swap_remove(i);
            if sz > want {
                self.free_list.push((off + want, sz - want));
            }
            self.allocated.insert(off, want);
            return off as i64;
        }
        // Bump a fresh block from the high-water mark and record it as a live allocation.
        match self.arena_bump(want) {
            Some(ptr) => {
                self.allocated.insert(ptr, want);
                ptr as i64
            }
            None => 0, // out of heap → NULL
        }
    }

    /// Bump `n` (already `ALIGN`-aligned) bytes off the heap high-water mark, returning the aligned
    /// start offset, or `None` if it would pass `heap_end`. The low-level arena primitive `malloc` and
    /// the `getenv` string cache both grow from — it advances `heap_next` but does **not** record an
    /// `allocated` entry (the caller decides whether the block is `free`-able).
    fn arena_bump(&mut self, n: u64) -> Option<u64> {
        let ptr = (self.heap_next + (ALIGN - 1)) & !(ALIGN - 1);
        match ptr.checked_add(n) {
            Some(end) if end <= self.heap_end => {
                self.heap_next = end;
                Some(ptr)
            }
            _ => None,
        }
    }

    /// `getcwd(buf, size) -> buf | 0`: copy the current directory (NUL-terminated, C `getcwd`'s
    /// contract) into the caller's window buffer; return `buf` on success, `-ERANGE` if the path plus
    /// its NUL won't fit `size`. `size == 0` with any path is `-EINVAL` (POSIX).
    fn getcwd(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let buf = *args.first().ok_or(Trap::Malformed)? as u64;
        let size = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        if size == 0 {
            return Ok(vec![EINVAL]);
        }
        let mut bytes = self.cwd.clone().into_bytes();
        bytes.push(0); // NUL terminator
        if bytes.len() as u64 > size {
            return Ok(vec![ERANGE]);
        }
        mem.write_bytes(buf, &bytes).ok_or(Trap::Malformed)?;
        Ok(vec![buf as i64])
    }

    /// `chdir(path, len) -> 0 | -errno`: set the working directory. The memfs is flat, so any UTF-8
    /// path is accepted as-is (no existence check — a follow-up, POSIX.md §6); a non-UTF-8 path is
    /// `-EINVAL`.
    fn chdir(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let plen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, plen).ok_or(Trap::Malformed)?;
        let Ok(path) = String::from_utf8(bytes) else {
            return Ok(vec![EINVAL]);
        };
        self.cwd = path;
        Ok(vec![0])
    }

    /// `getenv(name, len) -> ptr | 0`: look up an environment variable and return a **stable** window
    /// pointer to a NUL-terminated copy of its value (C `getenv`'s `char*` into libc storage), or `0`
    /// (C `NULL`) if unset. The copy is materialized in the arena once and cached (`env_ptrs`), so a
    /// repeated lookup returns the same pointer; `0` (out of heap) if the arena can't hold it. A
    /// non-UTF-8 name is treated as unset (`0`).
    fn getenv(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let ptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let nlen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let bytes = mem.read_bytes(ptr, nlen).ok_or(Trap::Malformed)?;
        let Ok(name) = String::from_utf8(bytes) else {
            return Ok(vec![0]); // a name we can't represent can't be set
        };
        if let Some(&cached) = self.env_ptrs.get(&name) {
            return Ok(vec![cached as i64]);
        }
        let Some(value) = self.env.get(&name).cloned() else {
            return Ok(vec![0]); // unset → NULL
        };
        let mut vb = value.into_bytes();
        vb.push(0); // NUL terminator
        let Some(dst) = self.arena_bump(vb.len() as u64) else {
            return Ok(vec![0]); // no room → behave as if unset (best effort)
        };
        mem.write_bytes(dst, &vb).ok_or(Trap::Malformed)?;
        self.env_ptrs.insert(name, dst);
        Ok(vec![dst as i64])
    }

    /// `setenv(name, nlen, value, vlen, overwrite) -> 0 | -errno`: set (or, when `overwrite == 0` and
    /// the name already exists, leave) an environment variable. Invalidates any cached `getenv` pointer
    /// for the name so the next `getenv` materializes the new value. A non-UTF-8 name/value is `-EINVAL`.
    fn setenv(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let nptr = *args.first().ok_or(Trap::Malformed)? as u64;
        let nlen = (*args.get(1).ok_or(Trap::Malformed)?).max(0) as u64;
        let vptr = *args.get(2).ok_or(Trap::Malformed)? as u64;
        let vlen = (*args.get(3).ok_or(Trap::Malformed)?).max(0) as u64;
        let overwrite = *args.get(4).ok_or(Trap::Malformed)?;
        let nb = mem.read_bytes(nptr, nlen).ok_or(Trap::Malformed)?;
        let vb = mem.read_bytes(vptr, vlen).ok_or(Trap::Malformed)?;
        let (Ok(name), Ok(value)) = (String::from_utf8(nb), String::from_utf8(vb)) else {
            return Ok(vec![EINVAL]);
        };
        if overwrite == 0 && self.env.contains_key(&name) {
            return Ok(vec![0]); // keep the existing value
        }
        self.env_ptrs.remove(&name); // stale cached pointer no longer reflects the value
        self.env.insert(name, value);
        Ok(vec![0])
    }

    /// `free(ptr)`: return `ptr`'s block to the free list for reuse. `free(NULL)` and a double / bogus
    /// free are no-ops (a bogus free never corrupts the arena — the size table is host-side). No
    /// coalescing yet (POSIX.md §6).
    fn free(&mut self, args: &[i64]) {
        let ptr = *args.first().unwrap_or(&0) as u64;
        if ptr == 0 {
            return;
        }
        if let Some(size) = self.allocated.remove(&ptr) {
            self.free_list.push((ptr, size));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use svm_interp::{run_capture_reserved_with_host, Host, Value};
    use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
    use svm_text::parse_module;
    use svm_verify::verify_module;

    const HEAP_BASE: u64 = 4096;
    const HEAP_END: u64 = 64 << 10;
    const WIN: usize = 128 << 10;

    /// func 0 `(host_fn_handle) -> i64`: `malloc(2)`, store `"hi"` into the returned buffer,
    /// `write(1, ptr, 2)`, then encode `write_result * 1_000_000 + ptr`. `malloc` hands out the aligned
    /// heap base (`4096`), `write` returns `2`, so the result is `2_004096` — and stdout is `"hi"`.
    const MALLOC_WRITE: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vph: i32):\n\
  vsz = i64.const 2\n\
  vptr = cap.call 13 2 (i64) -> (i64) vph (vsz)\n\
  vh = i32.const 104\n\
  i32.store8 vptr vh\n\
  vone = i64.const 1\n\
  vp1 = i64.add vptr vone\n\
  vi = i32.const 105\n\
  i32.store8 vp1 vi\n\
  vfd = i64.const 1\n\
  vn = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd, vptr, vsz)\n\
  vk = i64.const 1000000\n\
  vt = i64.mul vn vk\n\
  vr = i64.add vt vptr\n\
  return vr\n\
}\n";

    fn run_interp(src: &str, stdin: &[u8]) -> (Result<Vec<Value>, svm_interp::Trap>, Vec<u8>) {
        let m = parse_module(src).expect("parse");
        verify_module(&m).expect("verify");
        let mut host = Host::new();
        let (h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, stdin.to_vec());
        let mut fuel = 5_000_000u64;
        let r = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut host,
        )
        .0;
        (r, posix.stdout())
    }

    fn run_jit(src: &str, stdin: &[u8]) -> (JitOutcome, Vec<u8>) {
        let m = parse_module(src).expect("parse");
        verify_module(&m).expect("verify");
        let mut host = Host::new();
        let (h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, stdin.to_vec());
        let jo = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[h as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut host as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;
        (jo, posix.stdout())
    }

    /// func 0 `(handle) -> i64`: `read(0, buf, 8)` into a `malloc`'d buffer, then `write(1, buf, n)` —
    /// a cat-style echo. Returns `n` (bytes read); stdout is whatever stdin held.
    const READ_ECHO: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vph: i32):\n\
  veight = i64.const 8\n\
  vbuf = cap.call 13 2 (i64) -> (i64) vph (veight)\n\
  vfd0 = i64.const 0\n\
  vn = cap.call 13 1 (i64, i64, i64) -> (i64) vph (vfd0, vbuf, veight)\n\
  vfd1 = i64.const 1\n\
  vw = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vbuf, vn)\n\
  return vn\n\
}\n";

    /// func 0 `(handle) -> i64`: `exit(42)` — never returns.
    const EXIT_42: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vph: i32):\n\
  vc = i64.const 42\n\
  vx = cap.call 13 4 (i64) -> (i64) vph (vc)\n\
  vz = i64.const 0\n\
  return vz\n\
}\n";

    #[test]
    fn read_echo_matches_across_backends() {
        let (ir, iout) = run_interp(READ_ECHO, b"cat\n");
        let (jo, jout) = run_jit(READ_ECHO, b"cat\n");
        assert_eq!(ir, Ok(vec![Value::I64(4)]), "interp: read 4 bytes of stdin");
        assert_eq!(iout, b"cat\n", "interp: echoed stdin to stdout");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[4]),
            "jit: read count must match interp, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: echoed output must match interp");
    }

    /// func 0 `(handle) -> i64`: `a = malloc(32)`, `b = malloc(32)`, `free(a)`, `c = malloc(32)`, then
    /// return `(c - a) * 1_000_000 + (b - a)`. A working free list reuses `a`'s exact block for `c`
    /// (`c - a == 0`), and `b` sits one 32-byte block above `a` (`b - a == 32`) → `32`. (Without reuse
    /// `c` would bump fresh to `a + 64`, giving `64_000032` — so the value is non-vacuous.)
    const MALLOC_FREE_REUSE: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vph: i32):\n\
  vsz = i64.const 32\n\
  va = cap.call 13 2 (i64) -> (i64) vph (vsz)\n\
  vb = cap.call 13 2 (i64) -> (i64) vph (vsz)\n\
  vf = cap.call 13 3 (i64) -> (i64) vph (va)\n\
  vc = cap.call 13 2 (i64) -> (i64) vph (vsz)\n\
  vcva = i64.sub vc va\n\
  vbva = i64.sub vb va\n\
  vk = i64.const 1000000\n\
  vt = i64.mul vcva vk\n\
  vr = i64.add vt vbva\n\
  return vr\n\
}\n";

    #[test]
    fn free_list_reuses_a_freed_block_on_both() {
        let (ir, _iout) = run_interp(MALLOC_FREE_REUSE, b"");
        let (jo, _jout) = run_jit(MALLOC_FREE_REUSE, b"");
        // c reused a (diff 0); b is 32 bytes above a → 0*1_000_000 + 32 = 32.
        assert_eq!(
            ir,
            Ok(vec![Value::I64(32)]),
            "interp: free then malloc reuses the block"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[32]),
            "jit: allocator must match interp, got {jo:?}"
        );
    }

    #[test]
    fn exit_terminates_on_both_backends() {
        let (ir, _iout) = run_interp(EXIT_42, b"");
        let (jo, _jout) = run_jit(EXIT_42, b"");
        assert_eq!(ir, Err(svm_interp::Trap::Exit(42)), "interp: exit(42)");
        assert!(
            matches!(jo, JitOutcome::Exited(42)),
            "jit: exit(42) must terminate the domain, got {jo:?}"
        );
    }

    #[test]
    fn malloc_store_write_matches_across_backends() {
        let (ir, iout) = run_interp(MALLOC_WRITE, b"");
        let (jo, jout) = run_jit(MALLOC_WRITE, b"");
        // Interpreter reference: malloc → 4096, write → 2, so 2*1_000_000 + 4096 = 2_004096; "hi" out.
        assert_eq!(
            ir,
            Ok(vec![Value::I64(2_004_096)]),
            "interp: malloc+write result"
        );
        assert_eq!(
            iout, b"hi",
            "interp: bytes written to stdout via the personality"
        );
        // JIT parity: the HostFn dispatches through the same Host path, so identical result + output.
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2_004_096]),
            "jit: must match interp, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: stdout must match interp");
    }

    /// func 0 `(handle) -> i64`: `open("f", O_CREAT|O_RDWR)`, `write` "Hi!", `lseek` to 0, `read` it
    /// back, echo the bytes to stdout. Returns `fd * 1_000_000 + read_count`. The first file fd is `3`
    /// and 3 bytes round-trip → `3_000003`; stdout and the memfs file `"f"` are both `"Hi!"`.
    const FILE_ROUNDTRIP: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vph: i32):\n\
  vpath = i64.const 0\n\
  vfch = i32.const 102\n\
  i32.store8 vpath vfch\n\
  vplen = i64.const 1\n\
  vflags = i64.const 66\n\
  vfd = cap.call 13 5 (i64, i64, i64) -> (i64) vph (vpath, vplen, vflags)\n\
  a16 = i64.const 16\n\
  cH = i32.const 72\n\
  i32.store8 a16 cH\n\
  a17 = i64.const 17\n\
  ci = i32.const 105\n\
  i32.store8 a17 ci\n\
  a18 = i64.const 18\n\
  cbang = i32.const 33\n\
  i32.store8 a18 cbang\n\
  vwlen = i64.const 3\n\
  vw = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd, a16, vwlen)\n\
  vzero = i64.const 0\n\
  vsk = cap.call 13 7 (i64, i64, i64) -> (i64) vph (vfd, vzero, vzero)\n\
  a32 = i64.const 32\n\
  veight = i64.const 8\n\
  vr = cap.call 13 1 (i64, i64, i64) -> (i64) vph (vfd, a32, veight)\n\
  vfd1 = i64.const 1\n\
  vso = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd1, a32, vr)\n\
  vk = i64.const 1000000\n\
  vt = i64.mul vfd vk\n\
  vres = i64.add vt vr\n\
  return vres\n\
}\n";

    #[test]
    fn file_open_write_seek_read_matches_across_backends() {
        // Interpreter.
        let mut ih = Host::new();
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, Vec::new());
        let m = parse_module(FILE_ROUNDTRIP).expect("parse");
        verify_module(&m).expect("verify");
        let mut fuel = 5_000_000u64;
        let ir = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut ih,
        )
        .0;
        // JIT.
        let mut jh = Host::new();
        let (jhh, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, Vec::new());
        let jo = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[jhh as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;

        // fd 3, 3 bytes read → 3_000003; the file and the echoed stdout both hold "Hi!".
        assert_eq!(
            ir,
            Ok(vec![Value::I64(3_000_003)]),
            "interp: file roundtrip"
        );
        assert_eq!(iposix.stdout(), b"Hi!", "interp: echoed the file's bytes");
        assert_eq!(
            iposix.read_file("f").as_deref(),
            Some(&b"Hi!"[..]),
            "interp: memfs file written"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[3_000_003]),
            "jit: file roundtrip must match interp, got {jo:?}"
        );
        assert_eq!(
            jposix.stdout(),
            b"Hi!",
            "jit: echoed bytes must match interp"
        );
        assert_eq!(
            jposix.read_file("f").as_deref(),
            Some(&b"Hi!"[..]),
            "jit: memfs file written"
        );
    }

    /// func 0 `(handle) -> i64`: `unlink("g")` (a preloaded file → `0`), then `open("g", O_RDONLY)` (now
    /// gone → `-ENOENT`). Returns `unlink_result * 1000 + (-open_result)` = `0*1000 + 2` = `2`.
    const UNLINK_THEN_OPEN: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vph: i32):\n\
  vpath = i64.const 0\n\
  vg = i32.const 103\n\
  i32.store8 vpath vg\n\
  vplen = i64.const 1\n\
  vu = cap.call 13 8 (i64, i64) -> (i64) vph (vpath, vplen)\n\
  vflags = i64.const 0\n\
  vo = cap.call 13 5 (i64, i64, i64) -> (i64) vph (vpath, vplen, vflags)\n\
  vzero = i64.const 0\n\
  vneg = i64.sub vzero vo\n\
  vk = i64.const 1000\n\
  vt = i64.mul vu vk\n\
  vr = i64.add vt vneg\n\
  return vr\n\
}\n";

    #[test]
    fn unlink_removes_then_open_is_enoent_on_both() {
        let m = parse_module(UNLINK_THEN_OPEN).expect("parse");
        verify_module(&m).expect("verify");
        let mut ih = Host::new();
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, Vec::new());
        iposix.write_file("g", b"x");
        let mut fuel = 5_000_000u64;
        let ir = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut ih,
        )
        .0;
        let mut jh = Host::new();
        let (jhh, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, Vec::new());
        jposix.write_file("g", b"x");
        let jo = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[jhh as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;
        assert_eq!(
            ir,
            Ok(vec![Value::I64(2)]),
            "interp: unlink 0, then open -ENOENT"
        );
        assert_eq!(iposix.read_file("g"), None, "interp: file is gone");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2]),
            "jit: must match interp, got {jo:?}"
        );
        assert_eq!(jposix.read_file("g"), None, "jit: file is gone");
    }

    /// func 0 `(handle) -> i64`: `getenv("PATH")` (name bytes staged at offset 0 by the harness), then
    /// `write(1, ptr, 4)` echoing the first 4 bytes of the value to stdout, and return the returned
    /// pointer. With `PATH=/bin` staged host-side, `getenv` materializes `"/bin\0"` in the arena at the
    /// heap base (`4096`) and returns it; stdout is `"/bin"`.
    const GETENV_ECHO: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vph: i32):\n\
  vp = i64.const 0\n\
  vP = i32.const 80\n\
  i32.store8 vp vP\n\
  vp1 = i64.const 1\n\
  vA = i32.const 65\n\
  i32.store8 vp1 vA\n\
  vp2 = i64.const 2\n\
  vT = i32.const 84\n\
  i32.store8 vp2 vT\n\
  vp3 = i64.const 3\n\
  vH = i32.const 72\n\
  i32.store8 vp3 vH\n\
  vnlen = i64.const 4\n\
  vptr = cap.call 13 11 (i64, i64) -> (i64) vph (vp, vnlen)\n\
  vfd1 = i64.const 1\n\
  vfour = i64.const 4\n\
  vw = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vptr, vfour)\n\
  return vptr\n\
}\n";

    #[test]
    fn getenv_returns_stable_ptr_and_value_on_both() {
        let m = parse_module(GETENV_ECHO).expect("parse");
        verify_module(&m).expect("verify");
        // Interpreter.
        let mut ih = Host::new();
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, Vec::new());
        iposix.set_env("PATH", "/bin");
        let mut fuel = 5_000_000u64;
        let ir = run_capture_reserved_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut ih,
        )
        .0;
        // JIT.
        let mut jh = Host::new();
        let (jhh, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, Vec::new());
        jposix.set_env("PATH", "/bin");
        let jo = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[jhh as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;
        // getenv materializes "/bin\0" at the aligned heap base (4096) and returns it.
        assert_eq!(
            ir,
            Ok(vec![Value::I64(HEAP_BASE as i64)]),
            "interp: getenv returns the arena pointer"
        );
        assert_eq!(iposix.stdout(), b"/bin", "interp: echoed the env value");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[HEAP_BASE as i64]),
            "jit: getenv pointer must match interp, got {jo:?}"
        );
        assert_eq!(
            jposix.stdout(),
            b"/bin",
            "jit: echoed value must match interp"
        );
    }

    /// func 0 `(handle) -> i64`: `chdir("/tmp")` (path bytes staged at offset 0), then `getcwd(buf, 8)`
    /// into a scratch window buffer, echo the result (minus its NUL) to stdout, and return
    /// `chdir_result * 1_000_000 + getcwd_ptr`. A working roundtrip: `chdir` → `0`, `getcwd` writes
    /// `"/tmp\0"` and returns the buffer offset (32) → `0*1_000_000 + 32 = 32`; stdout is `"/tmp"`.
    const CHDIR_GETCWD: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vph: i32):\n\
  vp = i64.const 0\n\
  vsl = i32.const 47\n\
  i32.store8 vp vsl\n\
  vp1 = i64.const 1\n\
  vt = i32.const 116\n\
  i32.store8 vp1 vt\n\
  vp2 = i64.const 2\n\
  vm = i32.const 109\n\
  i32.store8 vp2 vm\n\
  vp3 = i64.const 3\n\
  vpc = i32.const 112\n\
  i32.store8 vp3 vpc\n\
  vplen = i64.const 4\n\
  vcd = cap.call 13 10 (i64, i64) -> (i64) vph (vp, vplen)\n\
  vbuf = i64.const 32\n\
  veight = i64.const 8\n\
  vgc = cap.call 13 9 (i64, i64) -> (i64) vph (vbuf, veight)\n\
  vfd1 = i64.const 1\n\
  vfour = i64.const 4\n\
  vw = cap.call 13 0 (i64, i64, i64) -> (i64) vph (vfd1, vbuf, vfour)\n\
  vk = i64.const 1000000\n\
  vtt = i64.mul vcd vk\n\
  vr = i64.add vtt vgc\n\
  return vr\n\
}\n";

    #[test]
    fn chdir_then_getcwd_roundtrips_on_both() {
        let (ir, iout) = run_interp(CHDIR_GETCWD, b"");
        let (jo, jout) = run_jit(CHDIR_GETCWD, b"");
        // chdir 0, getcwd returns buf (32) → 0*1_000_000 + 32 = 32; stdout "/tmp".
        assert_eq!(
            ir,
            Ok(vec![Value::I64(32)]),
            "interp: chdir then getcwd roundtrip"
        );
        assert_eq!(iout, b"/tmp", "interp: getcwd wrote the new cwd");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[32]),
            "jit: roundtrip must match interp, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: getcwd output must match interp");
    }

    #[test]
    fn setenv_updates_and_getenv_repoints_at_the_new_value() {
        // A host-level unit for the setenv/getenv cache-invalidation contract (no guest module needed):
        // getenv caches a pointer; setenv must invalidate it so the next getenv reflects the new value.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        let mut st = posix.inner.lock().unwrap();
        // Stage the name "K" at offset 0 and value "v2" at offset 8 in a scratch window.
        let mut win = vec![0u8; WIN];
        win[0] = b'K';
        win[8] = b'v';
        win[9] = b'2';
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
        // setenv("K", "v1", overwrite=1): name@0 len 1, value staged separately — reuse offset 8 with "v1".
        st.env.insert("K".to_string(), "v1".to_string());
        // getenv("K") materializes "v1\0" and caches the pointer.
        let p1 = st.getenv(&[0, 1], Some(&mut mem)).unwrap()[0];
        assert!(p1 > 0, "getenv returns a non-null arena pointer");
        // setenv("K", "v2", overwrite=1): name@0 len1, value@8 len2.
        let r = st.setenv(&[0, 1, 8, 2, 1], Some(&mut mem)).unwrap()[0];
        assert_eq!(r, 0, "setenv succeeds");
        // getenv("K") now re-materializes at a *fresh* pointer holding "v2\0".
        let p2 = st.getenv(&[0, 1], Some(&mut mem)).unwrap()[0];
        assert_ne!(p2, p1, "setenv invalidated the cached getenv pointer");
        let got = mem.read_bytes(p2 as u64, 3).unwrap();
        assert_eq!(got, b"v2\0", "getenv reflects the setenv'd value");
        // overwrite=0 on an existing name is a no-op (keeps "v2").
        let r0 = st.setenv(&[0, 1, 8, 2, 0], Some(&mut mem)).unwrap()[0];
        assert_eq!(r0, 0, "setenv(overwrite=0) on existing name returns 0");
        assert_eq!(
            st.env.get("K").map(String::as_str),
            Some("v2"),
            "overwrite=0 kept the existing value"
        );
    }

    #[test]
    fn stat_and_readdir_over_the_memfs() {
        // A host-level unit for the fs-metadata surface (stat + the opendir/readdir/closedir stream):
        // a file stats as a regular file with its size; a path with children stats as a directory and
        // enumerates its immediate children (files *and* the subdir once), sorted, ending at `0`.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        posix.write_file("/tmp/a", b"hello");
        posix.write_file("/tmp/b", b"hi");
        posix.write_file("/tmp/sub/c", b"x");

        let mut win = vec![0u8; WIN];
        win[..6].copy_from_slice(b"/tmp/a"); // path at offset 0
        win[100..104].copy_from_slice(b"/tmp"); // dir path at offset 100
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
        let mut st = posix.inner.lock().unwrap();
        let rd = |mem: &svm_interp::WindowMem, off: u64| {
            i64::from_le_bytes(mem.read_bytes(off, 8).unwrap().try_into().unwrap())
        };

        // stat("/tmp/a", statbuf@200) → regular file, size 5.
        assert_eq!(st.stat(&[0, 6, 200], Some(&mut mem)).unwrap()[0], 0);
        assert_eq!(rd(&mem, 200), S_IFREG | 0o644, "st_mode: regular file");
        assert_eq!(rd(&mem, 208), 5, "st_size: the file's byte length");

        // stat("/tmp") → directory.
        assert_eq!(st.stat(&[100, 4, 200], Some(&mut mem)).unwrap()[0], 0);
        assert_eq!(rd(&mem, 200), S_IFDIR | 0o755, "st_mode: directory");

        // stat of an absent path → -ENOENT.
        win_write(&mut mem, 400, b"/nope");
        assert_eq!(st.stat(&[400, 5, 200], Some(&mut mem)).unwrap()[0], ENOENT);

        // opendir("/tmp") → children {a, b, sub} (the subdir listed once), sorted, then `0` at end.
        let dir = st.opendir(&[100, 4], Some(&mut mem)).unwrap()[0];
        assert!(dir >= 0, "opendir returns a stream handle");
        let mut got = Vec::new();
        loop {
            let n = st.readdir(&[dir, 300, 64], Some(&mut mem)).unwrap()[0];
            if n == 0 {
                break;
            }
            got.push(String::from_utf8(mem.read_bytes(300, n as u64).unwrap()).unwrap());
        }
        assert_eq!(got, vec!["a", "b", "sub"], "immediate children, sorted");
        assert_eq!(st.closedir(&[dir]), 0);
        assert_eq!(st.closedir(&[dir]), EBADF, "double closedir is -EBADF");

        // opendir of a regular file → -ENOTDIR.
        assert_eq!(st.opendir(&[0, 6], Some(&mut mem)).unwrap()[0], ENOTDIR);
    }

    /// Write `bytes` into `mem` at `off` (test helper — `WindowMem` has no direct slice setter).
    fn win_write(mem: &mut svm_interp::WindowMem, off: u64, bytes: &[u8]) {
        mem.write_bytes(off, bytes).unwrap();
    }

    #[test]
    fn argc_argv_deliver_the_argument_vector() {
        // The host-side argument vector (the `sh -c "…"` path): `argc` reports the count, `argv(i, …)`
        // writes arg `i` NUL-terminated; an out-of-range index is -EINVAL.
        let mut host = Host::new();
        let (_h, posix) = grant(&mut host, HEAP_BASE, HEAP_END, Vec::new());
        posix.set_args(&["sh", "-c", "echo hi"]);
        let mut win = vec![0u8; WIN];
        let mut mem = svm_interp::WindowMem::new(&mut win, WIN as u64);
        let mut st = posix.inner.lock().unwrap();

        assert_eq!(st.args.len() as i64, 3, "argc");
        assert_eq!(st.argv(&[1, 0, 64], Some(&mut mem)).unwrap()[0], 2); // "-c" len 2
        assert_eq!(mem.read_bytes(0, 3).unwrap(), b"-c\0");
        assert_eq!(st.argv(&[2, 100, 64], Some(&mut mem)).unwrap()[0], 7); // "echo hi"
        assert_eq!(mem.read_bytes(100, 8).unwrap(), b"echo hi\0");
        assert_eq!(st.argv(&[9, 0, 64], Some(&mut mem)).unwrap()[0], EINVAL);
    }

    #[test]
    fn resolve_binds_libc_names() {
        // The §7 name → (HOST_FN, op) map a linker uses to bind a shell's libc imports.
        assert_eq!(resolve("malloc").map(|c| c.op), Some(OP_MALLOC));
        assert_eq!(resolve("posix.write").map(|c| c.op), Some(OP_WRITE));
        assert_eq!(resolve("_exit").map(|c| c.op), Some(OP_EXIT));
        assert_eq!(resolve("open").map(|c| c.op), Some(OP_OPEN));
        assert_eq!(resolve("lseek").map(|c| c.op), Some(OP_LSEEK));
        assert!(
            resolve("dlopen").is_none(),
            "unknown libc name fails closed"
        );
    }

    /// The **real linking path**: a module that *imports* the libc names `malloc`/`write` (never
    /// hand-writes a `cap.call`), exactly what a chibicc/`svm-llvm` frontend emits for unresolved libc
    /// symbols. `svm_ir::resolve_imports` binds each name through [`resolve`] and lowers every
    /// `call.import` to a `cap.call` on the personality's handle — the same program, now import-free,
    /// runs identically on both backends. Semantically identical to `MALLOC_WRITE` (→ `2_004096`, `"hi"`),
    /// so it proves the *binding*, not new behavior.
    const IMPORT_MALLOC_WRITE: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block0(vph: i32):\n\
  vsz = i64.const 2\n\
  vptr = call.import \"malloc\" (i64) -> (i64) vph (vsz)\n\
  vh = i32.const 104\n\
  i32.store8 vptr vh\n\
  vone = i64.const 1\n\
  vp1 = i64.add vptr vone\n\
  vi = i32.const 105\n\
  i32.store8 vp1 vi\n\
  vfd = i64.const 1\n\
  vn = call.import \"write\" (i64, i64, i64) -> (i64) vph (vfd, vptr, vsz)\n\
  vk = i64.const 1000000\n\
  vt = i64.mul vn vk\n\
  vr = i64.add vt vptr\n\
  return vr\n\
}\n";

    /// The §7 **general form**: the module's imports carry a `ConstI32` **placeholder** handle
    /// (`vph = i32.const 0`) and the entry takes **no capability parameters at all** — the granted
    /// handle arrives by [`resolve_bound`] patching the placeholder at resolve (`Resolved::CapBound`),
    /// never through an entry argument or a powerbox slot. Same program as `IMPORT_MALLOC_WRITE`
    /// (→ `2_004096`, `"hi"`), differing only in how the authority binds.
    const IMPORT_BOUND_MALLOC_WRITE: &str = "memory 17\n\
func () -> (i64) {\n\
block0():\n\
  vph = i32.const 0\n\
  vsz = i64.const 2\n\
  vptr = call.import \"malloc\" (i64) -> (i64) vph (vsz)\n\
  vh = i32.const 104\n\
  i32.store8 vptr vh\n\
  vone = i64.const 1\n\
  vp1 = i64.add vptr vone\n\
  vi = i32.const 105\n\
  i32.store8 vp1 vi\n\
  vph2 = i32.const 0\n\
  vfd = i64.const 1\n\
  vn = call.import \"write\" (i64, i64, i64) -> (i64) vph2 (vfd, vptr, vsz)\n\
  vk = i64.const 1000000\n\
  vt = i64.mul vn vk\n\
  vr = i64.add vt vptr\n\
  return vr\n\
}\n";

    #[test]
    fn bound_imports_supply_the_handle_at_resolve() {
        let m = parse_module(IMPORT_BOUND_MALLOC_WRITE).expect("parse");

        // Grant FIRST (resolution needs the handle), on two identical hosts; deterministic grant
        // order gives both backends the same handle value, so one resolved module serves both.
        let mut ih = Host::new();
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, Vec::new());
        let mut jh = Host::new();
        let (jhh, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, Vec::new());
        assert_eq!(h, jhh, "identical grant order → identical handle");

        let resolved =
            svm_ir::resolve_imports_with(&m, resolve_bound(h)).expect("bound imports resolve");
        assert!(resolved.imports.is_empty(), "resolution is import-free");
        verify_module(&resolved).expect("verify the resolved module");

        // No entry args: the program holds no capability parameters — authority came in at resolve.
        let mut fuel = 5_000_000u64;
        let ir =
            run_capture_reserved_with_host(&resolved, 0, &[], &mut fuel, &[0u8; WIN], 0, &mut ih).0;
        let jo = compile_and_run_capture_reserved_with_host(
            &resolved,
            0,
            &[],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;

        assert_eq!(
            ir,
            Ok(vec![Value::I64(2_004_096)]),
            "interp: bound-handle malloc+write"
        );
        assert_eq!(
            iposix.stdout(),
            b"hi",
            "interp: the write reached the personality"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2_004_096]),
            "jit: must match interp, got {jo:?}"
        );
        assert_eq!(jposix.stdout(), b"hi", "jit: stdout must match interp");
    }

    #[test]
    fn named_imports_bind_through_resolve_and_run() {
        let m = parse_module(IMPORT_MALLOC_WRITE).expect("parse");
        assert_eq!(
            m.imports
                .iter()
                .map(|i| i.name.as_str())
                .collect::<Vec<_>>(),
            ["malloc", "write"],
            "the module declares the libc names it imports"
        );
        // §7 late binding: bind each import name through the personality's resolver, lowering
        // `call.import` → `cap.call` on the handle operand. Fails closed on an unknown name.
        let resolved = svm_ir::resolve_imports(&m, resolve).expect("all libc imports resolve");
        assert!(
            resolved.imports.is_empty(),
            "resolution drops the import section — the result is import-free"
        );
        verify_module(&resolved).expect("verify the resolved module");

        // Run the resolved (import-free) module on both backends with the personality granted.
        let mut ih = Host::new();
        let (h, iposix) = grant(&mut ih, HEAP_BASE, HEAP_END, Vec::new());
        let mut fuel = 5_000_000u64;
        let ir = run_capture_reserved_with_host(
            &resolved,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[0u8; WIN],
            0,
            &mut ih,
        )
        .0;
        let mut jh = Host::new();
        let (jh_handle, jposix) = grant(&mut jh, HEAP_BASE, HEAP_END, Vec::new());
        let jo = compile_and_run_capture_reserved_with_host(
            &resolved,
            0,
            &[jh_handle as i64],
            &[0u8; WIN],
            0,
            svm_run::cap_thunk,
            &mut jh as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit")
        .0;

        assert_eq!(
            ir,
            Ok(vec![Value::I64(2_004_096)]),
            "interp: malloc+write through bound imports"
        );
        assert_eq!(
            iposix.stdout(),
            b"hi",
            "interp: personality captured the write"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[2_004_096]),
            "jit: bound-import run must match interp, got {jo:?}"
        );
        assert_eq!(jposix.stdout(), b"hi", "jit: stdout must match interp");
    }
}
