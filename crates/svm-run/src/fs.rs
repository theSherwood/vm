//! **Configurable filesystem capability backends** — §7 embedder-registered `HostFn`s, *not* part of
//! the default powerbox grant. A consumer that wants its guest to see a filesystem offers one
//! explicitly (wasm-style dependency injection): grant it under a name via
//! [`Instance::run_with_caps`](crate::Instance::run_with_caps) (or bind it to an import with
//! [`instantiate_with_imports`](crate::instantiate_with_imports)), and the guest reaches it with
//! `__vm_cap_resolve("fs")` + `__vm_host_call(handle, op, …)`. No filesystem authority exists unless
//! the embedder injects it, and the two backends here are interchangeable behind the same op protocol:
//!
//! - [`mem_fs`] — a deterministic **in-memory** filesystem (fresh per run). The hermetic default for
//!   tests and differential runs: no real-fs state, no cleanup, parallel-safe.
//! - [`host_fs`] — the **real** filesystem, attenuated to a root directory: the capability *is* the
//!   rooted directory (relative paths only; `..` and absolute paths are refused), so the guest cannot
//!   name anything outside it.
//!
//! ## Op protocol (`__vm_host_call(handle, op, a, b, c, d) -> i64`)
//!
//! | op | name | args | returns |
//! |----|------|------|---------|
//! | 0 | `open` | `(path_ptr, path_len, flags, _)` | fd ≥ 0 |
//! | 1 | `read` | `(fd, buf_ptr, len, _)` | bytes read (0 = EOF) |
//! | 2 | `write` | `(fd, buf_ptr, len, _)` | bytes written |
//! | 3 | `seek` | `(fd, whence 0/1/2, offset, _)` | new position |
//! | 4 | `close` | `(fd, _, _, _)` | 0 |
//! | 5 | `remove` | `(path_ptr, path_len, _, _)` | 0 |
//! | 6 | `rename` | `(from_ptr, from_len, to_ptr, to_len)` | 0 |
//! | 7 | `truncate` | `(fd, new_len, _, _)` | 0 |
//! | 8 | `sync` | `(fd, _, _, _)` | 0 |
//! | 9 | `mmap` | `(fd, file_offset, len, win_buf)` | 0 |
//! | 10 | `msync` | `(win_buf, len, _, _)` | 0 |
//! | 11 | `munmap` | `(win_buf, _, _, _)` | 0 |
//! | 12 | `crash_arm` | `(n, _, _, _)` | 0 / `-EINVAL` |
//! | 13 | `map_region` | `(fd, file_offset, len, _)` | region handle / `-errno` |
//! | 14 | `stat` | `(path_ptr, path_len, statbuf_ptr, statbuf_cap)` | 0 / `-errno` |
//! | 15 | `mkdir` | `(path_ptr, path_len, _, _)` | 0 / `-errno` |
//! | 16 | `rmdir` | `(path_ptr, path_len, _, _)` | 0 / `-errno` |
//! | 17 | `opendir` | `(path_ptr, path_len, _, _)` | dir handle ≥ 0 / `-errno` |
//! | 18 | `readdir` | `(dh, name_ptr, name_cap, _)` | name length > 0, `0` = end, `-errno` |
//! | 19 | `closedir` | `(dh, _, _, _)` | 0 / `-errno` |
//!
//! Op 12 (`crash_arm`) exists **only on the `*_crashy` test variants** (see [`FS_CRASH_ARM`]); op 13
//! (`map_region`, the §4b zero-copy path — see [`FS_MAP_REGION`]) exists **only on [`host_fs_mmap`]**.
//! The default [`mem_fs`]/[`host_fs`] return `-EINVAL` for both (unknown op / no minting authority).
//!
//! ## The metadata + directory surface (`stat`/`mkdir`/`rmdir`/`opendir`/`readdir`/`closedir`)
//!
//! The per-access VFS above names *files*; a real data tree (a natively-`initdb`'d Postgres cluster,
//! say) also needs to be **walked**. `stat` fills a fixed 72-byte little-endian [`StatBuf`] (mode with
//! the `S_IF*` type bits, size, mtime, ino/dev, …) so the guest libc can tell a directory from a file
//! and read a size without opening; `mkdir`/`rmdir` create and remove directories; `opendir` snapshots
//! a directory's immediate entries and `readdir` yields their names one per call (`0` when exhausted),
//! `closedir` drops the handle. `stat` uses **lstat** semantics (does not follow symlinks) so a symlink
//! in the tree cannot be used to probe the file *type* of something outside the granted root. Both
//! backends implement the identical protocol; [`mem_fs`] models directories over its flat name table
//! (a path is a directory if it was `mkdir`'d or is a strict prefix of an existing file), so a
//! differential still runs identically on `mem_fs` and `host_fs`.
//!
//! ## The file-backed-mmap surface (`mmap`/`msync`/`munmap`) — the second storage shape
//!
//! `mmap` binds a **guest-owned window buffer** (`win_buf`, `len`) to a file region (`fd`,
//! `file_offset`): the host copies the file bytes *into* the buffer and records the binding. The
//! guest then reads and writes those bytes with ordinary loads/stores — **zero host calls on the
//! data-access path** (that is what makes this the memory-mapped shape, distinct from the per-access
//! `read`/`write` VFS). `msync(win_buf, len)` flushes a sub-range of a mapping back to its file at
//! `file_offset + (win_buf − mapping.base)`; `munmap` flushes the whole mapping and drops it. This is
//! coherent for a single mapping of a file (the buffer is the sole authority), exactly what an
//! `MDB_WRITEMAP`-mode LMDB needs: it writes every page — data and meta — through the map and asks
//! for durability via `msync`. (Not multi-process shared-memory coherence — that is a later slice.)
//!
//! Errors are negative errno values ([`ENOENT`]/[`EBADF`]/[`EINVAL`]/[`EACCES`]/[`EFAULT`]). `flags`
//! is a bitset ([`O_READ`]/[`O_WRITE`]/[`O_APPEND`]/[`O_TRUNC`]/[`O_CREATE`]) the guest libc derives
//! from the C `fopen` mode string. Buffers/paths are window-relative; an out-of-window range is
//! `-EFAULT` (fail-closed, never a host-side OOB).

use crate::HostCap;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use svm_interp::{GuestMem, HostFn, HostFnRegion, RegionMinter};

pub const FS_OPEN: u32 = 0;
pub const FS_READ: u32 = 1;
pub const FS_WRITE: u32 = 2;
pub const FS_SEEK: u32 = 3;
pub const FS_CLOSE: u32 = 4;
pub const FS_REMOVE: u32 = 5;
pub const FS_RENAME: u32 = 6;
pub const FS_TRUNCATE: u32 = 7;
pub const FS_SYNC: u32 = 8;
pub const FS_MMAP: u32 = 9;
pub const FS_MSYNC: u32 = 10;
pub const FS_MUNMAP: u32 = 11;
/// `map_region(fd, file_offset, len)` — the **zero-copy** file-mmap path (§4b of `MMAP_CAPABILITY.md`).
/// Mint a file-backed `SharedRegion` over `[file_offset, file_offset+len)` of the open file `fd` and
/// return its **handle** (a `SharedRegion` cap the guest maps into its window with `SharedRegion.map`),
/// so guest loads/stores hit the real file's pages with no copy-in and no per-access host call. Present
/// ONLY on the `host_fs_mmap` variant (which is granted with region-minting authority); returns
/// `-EINVAL` on `mem_fs`/`host_fs` (no minter, or no real fd). v1 requires `file_offset == 0`.
pub const FS_MAP_REGION: u32 = 13;
/// `crash_arm(n)` — **test-only** crash injection (§4d of `MMAP_CAPABILITY.md`). Present ONLY on the
/// `*_crashy` backend variants; the default [`mem_fs`]/[`host_fs`] leave the controller absent so this
/// op is an unknown op (`-EINVAL`) on a shipping grant. Arms a simulated power loss: after `n` further
/// durability barriers (`msync`/`sync`) have completed, the *next* barrier "crashes" — from then on
/// every write to the backing store is silently dropped (the un-synced page cache is lost, as on real
/// power loss) while **reads keep working** (a dead process's file is still readable on reopen). `n < 0`
/// disarms. Lets a test sweep the crash point across every sync boundary and prove the mapped store
/// recovers to its last *committed* state at each one.
pub const FS_CRASH_ARM: u32 = 12;

/// `stat(path_ptr, path_len, statbuf_ptr, statbuf_cap)` — fill a fixed [`StatBuf`] (72 bytes,
/// little-endian) for `path` with **lstat** semantics (symlinks are not followed). `statbuf_cap`
/// must be ≥ [`STATBUF_LEN`]. Returns `0` / `-errno`.
pub const FS_STAT: u32 = 14;
/// `mkdir(path_ptr, path_len)` — create a directory. `-EEXIST` if it already exists. (The `mode`
/// argument a guest `mkdir(2)` would pass is ignored: the granted root's umask governs.)
pub const FS_MKDIR: u32 = 15;
/// `rmdir(path_ptr, path_len)` — remove an empty directory (`-ENOTEMPTY` otherwise).
pub const FS_RMDIR: u32 = 16;
/// `opendir(path_ptr, path_len)` — open a directory for iteration; returns a **dir handle** (a small
/// non-negative integer, a separate namespace from file `fd`s) or `-errno`. The directory's immediate
/// entries are snapshotted at this call, so a concurrent create/remove does not perturb the walk
/// (matches a typical libc `readdir` buffering the getdents stream).
pub const FS_OPENDIR: u32 = 17;
/// `readdir(dh, name_ptr, name_cap)` — write the next entry's name (no trailing NUL) into the guest
/// buffer and return its byte length; `0` when the directory is exhausted; `-errno` on a bad handle
/// or `-EINVAL` if `name_cap` is too small for the next name. `.` and `..` are **not** yielded (the
/// guest libc synthesizes them if it wants them), matching what Postgres's `ReadDir` filters anyway.
pub const FS_READDIR: u32 = 18;
/// `closedir(dh)` — drop a dir handle opened by [`FS_OPENDIR`].
pub const FS_CLOSEDIR: u32 = 19;

/// Byte length of the fixed [`StatBuf`] the [`FS_STAT`] op writes.
pub const STATBUF_LEN: usize = 72;
/// `S_IFMT` mask and the two `S_IF*` type values the guest libc needs to tell files from directories
/// (Linux ABI values, so the guest shim can copy `mode` straight into a `struct stat`).
pub const S_IFMT: u32 = 0o170000;
pub const S_IFREG: u32 = 0o100000;
pub const S_IFDIR: u32 = 0o040000;
pub const S_IFLNK: u32 = 0o120000;

pub const O_READ: i64 = 1;
pub const O_WRITE: i64 = 2;
pub const O_APPEND: i64 = 4;
pub const O_TRUNC: i64 = 8;
pub const O_CREATE: i64 = 16;

pub const ENOENT: i64 = 2;
pub const EBADF: i64 = 9;
pub const EACCES: i64 = 13;
pub const EFAULT: i64 = 14;
pub const EEXIST: i64 = 17;
pub const ENOTDIR: i64 = 20;
pub const EINVAL: i64 = 22;
pub const ENOTEMPTY: i64 = 39;

/// Build the fixed 72-byte little-endian [`StatBuf`] payload from the fields the guest libc reads.
/// Layout (offset: field): `0:mode(u32) 4:nlink(u32) 8:size(i64) 16:mtime_sec(i64) 24:mtime_nsec(i64)
/// 32:ino(u64) 40:dev(u64) 48:uid(u32) 52:gid(u32) 56:blksize(i64) 64:blocks(i64)`.
#[allow(clippy::too_many_arguments)]
fn stat_bytes(
    mode: u32,
    nlink: u32,
    size: i64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ino: u64,
    dev: u64,
    uid: u32,
    gid: u32,
    blksize: i64,
    blocks: i64,
) -> [u8; STATBUF_LEN] {
    let mut b = [0u8; STATBUF_LEN];
    b[0..4].copy_from_slice(&mode.to_le_bytes());
    b[4..8].copy_from_slice(&nlink.to_le_bytes());
    b[8..16].copy_from_slice(&size.to_le_bytes());
    b[16..24].copy_from_slice(&mtime_sec.to_le_bytes());
    b[24..32].copy_from_slice(&mtime_nsec.to_le_bytes());
    b[32..40].copy_from_slice(&ino.to_le_bytes());
    b[40..48].copy_from_slice(&dev.to_le_bytes());
    b[48..52].copy_from_slice(&uid.to_le_bytes());
    b[52..56].copy_from_slice(&gid.to_le_bytes());
    b[56..64].copy_from_slice(&blksize.to_le_bytes());
    b[64..72].copy_from_slice(&blocks.to_le_bytes());
    b
}

/// Read a guest path (window `ptr`/`len`) as UTF-8. `-EFAULT` on an out-of-window range, `-EINVAL`
/// on non-UTF-8 or an unreasonable length, `-EACCES` on a path that could name anything outside the
/// granted root (absolute, `..`, or empty) — enforced by **both** backends so the protocol semantics
/// are backend-independent (a differential runs identically on `mem_fs` and `host_fs`).
fn read_path(mem: Option<&dyn GuestMem>, ptr: i64, len: i64) -> Result<String, i64> {
    let mem = mem.ok_or(-EFAULT)?;
    if !(0..=4096).contains(&len) || ptr < 0 {
        return Err(-EINVAL);
    }
    let bytes = mem.read_bytes(ptr as u64, len as u64).ok_or(-EFAULT)?;
    let path = String::from_utf8(bytes).map_err(|_| -EINVAL)?;
    let p = Path::new(&path);
    if path.is_empty()
        || p.is_absolute()
        || p.components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(-EACCES);
    }
    Ok(path)
}

/// Allocate a file descriptor, **reserving 0/1/2** (the POSIX stdin/stdout/stderr slots) so files
/// always start at 3. A guest libc that routes fds 0/1/2 to the powerbox `Stream` cap (stdout/stderr/
/// stdin) and everything else to this fs cap then never confuses a file fd with a stream fd — the two
/// namespaces are disjoint. The reserved slots stay permanently vacant (an op on fd 0/1/2 is `-EBADF`).
fn alloc_fd<T>(open: &mut Vec<Option<T>>) -> usize {
    const RESERVED: usize = 3;
    while open.len() < RESERVED {
        open.push(None);
    }
    match open.iter().skip(RESERVED).position(Option::is_none) {
        Some(off) => RESERVED + off,
        None => {
            open.push(None);
            open.len() - 1
        }
    }
}

/// One open file: a shared byte buffer (kept alive independently of the name table, so a `remove`
/// of an open file behaves POSIX-like — the data survives until the last close) + cursor + mode.
struct MemOpen {
    data: Arc<Mutex<Vec<u8>>>,
    pos: usize,
    readable: bool,
    writable: bool,
    append: bool,
}

/// Test-only crash-injection controller (the §4d "crash hook"), shared by both backends. Models a
/// power loss: [`FS_CRASH_ARM`] sets `countdown` to the number of durability barriers
/// (`msync`/`sync`) that may still complete; each barrier decrements it, and the one that finds it at
/// zero *trips* — sets `crashed`, and is itself dropped (the crash happened before it reached the
/// platter). Once `crashed`, every persisting op (`msync`/`sync`/`munmap` flush/`write`/`truncate`)
/// silently drops its effect, so the backing file is frozen at the last completed barrier; reads are
/// untouched. Present only on the `*_crashy` variants — a shipping grant has no controller at all.
#[derive(Default)]
struct CrashCtl {
    /// Barriers that may still complete before the crash trips; `None` = disarmed (never crash).
    countdown: Option<u64>,
    /// Once set, all persistence is frozen.
    crashed: bool,
}

impl CrashCtl {
    /// Call at each durability barrier (`msync`/`sync`). Returns `true` if this barrier's write must be
    /// **dropped** — either we have already crashed, or this very barrier trips the crash.
    fn barrier(&mut self) -> bool {
        if self.crashed {
            return true;
        }
        match self.countdown {
            Some(0) => {
                self.crashed = true;
                true // the crash struck mid-barrier: its bytes never reach the file
            }
            Some(n) => {
                self.countdown = Some(n - 1);
                false
            }
            None => false,
        }
    }
}

/// One live `mmap`: a guest window buffer `[base, base+len)` bound to `data` at `file_off`. The
/// guest reads/writes the window bytes directly; `msync` copies a sub-range back into `data`.
struct MemMapping {
    base: u64,
    len: u64,
    data: Arc<Mutex<Vec<u8>>>,
    file_off: u64,
}

#[derive(Default)]
struct MemFsState {
    files: HashMap<String, Arc<Mutex<Vec<u8>>>>,
    /// Explicitly-`mkdir`'d directories (normalized keys). A path is *also* treated as a directory
    /// when it is a strict prefix of an existing file key, so `initdb`-style trees created purely by
    /// writing files still walk correctly; `dirs` records the empty ones a walk would otherwise miss.
    dirs: std::collections::BTreeSet<String>,
    open: Vec<Option<MemOpen>>,
    /// Snapshots taken by [`FS_OPENDIR`]: `opendirs[dh]` is the remaining child names to yield.
    opendirs: Vec<Option<Vec<String>>>,
    maps: Vec<MemMapping>,
    /// `Some` only on the `mem_fs_crashy` variant (test-only crash injection); `None` on `mem_fs`.
    crash: Option<CrashCtl>,
}

/// Normalize a vetted relative path to a canonical key: drop `.`/empty segments, join with `/`.
/// The root (`.` or the effect of stripping everything) maps to `""`.
fn norm(p: &str) -> String {
    p.split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect::<Vec<_>>()
        .join("/")
}

impl MemFsState {
    /// Is `key` (already normalized) a directory in this flat store?
    fn is_dir(&self, key: &str) -> bool {
        key.is_empty() || self.dirs.contains(key) || {
            let prefix = format!("{key}/");
            self.files.keys().any(|k| k.starts_with(&prefix))
                || self.dirs.iter().any(|d| d.starts_with(&prefix))
        }
    }
    /// Immediate child names of directory `key` (normalized), deduplicated, sorted for determinism.
    fn children_of(&self, key: &str) -> Vec<String> {
        let prefix = if key.is_empty() {
            String::new()
        } else {
            format!("{key}/")
        };
        let mut set = std::collections::BTreeSet::new();
        let child = |full: &str| -> Option<String> {
            let rest = full.strip_prefix(&prefix)?;
            if rest.is_empty() {
                return None;
            }
            Some(rest.split('/').next().unwrap().to_string())
        };
        for k in self.files.keys() {
            if let Some(c) = child(k) {
                set.insert(c);
            }
        }
        for d in &self.dirs {
            if let Some(c) = child(d) {
                set.insert(c);
            }
        }
        set.into_iter().collect()
    }
}

impl MemFsState {
    /// A durability barrier (`msync`/`sync`): `true` ⇒ drop this write (crashed or crashing now).
    fn crash_barrier(&mut self) -> bool {
        self.crash.as_mut().is_some_and(CrashCtl::barrier)
    }
    /// Whether the backing store is frozen by a tripped crash (persisting ops become no-ops).
    fn crash_frozen(&self) -> bool {
        self.crash.as_ref().is_some_and(|c| c.crashed)
    }
}

impl MemFsState {
    fn handle(&mut self, op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>) -> i64 {
        let mut mem = mem;
        let a = |i: usize| args.get(i).copied().unwrap_or(0);
        match op {
            FS_OPEN => {
                let path = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let flags = a(2);
                let data = match self.files.get(&path) {
                    Some(d) => {
                        if flags & O_TRUNC != 0 {
                            d.lock().unwrap_or_else(|e| e.into_inner()).clear();
                        }
                        d.clone()
                    }
                    None => {
                        // A **read-only** open of an explicitly-tracked directory (`mkdir`'d or seeded)
                        // — e.g. Postgres `fsync`s directories at checkpoint via `open(dir, O_RDONLY)` +
                        // `fsync`. Return a read-only fd over an empty buffer: `sync`/`close` succeed,
                        // reads yield EOF, writes are refused (matches a real directory fd). Narrow by
                        // design: only a read-only open (a write/create never resolves to a dir), and
                        // only `dirs` (not the `""` root or a mere file-key prefix), so an ordinary
                        // `open(name, "w")` that happens to prefix another file still creates a file.
                        let write_intent = flags & (O_CREATE | O_WRITE | O_APPEND | O_TRUNC) != 0;
                        if !write_intent && self.dirs.contains(&path) {
                            let o = MemOpen {
                                data: Arc::new(Mutex::new(Vec::new())),
                                pos: 0,
                                readable: true,
                                writable: false,
                                append: false,
                            };
                            let fd = alloc_fd(&mut self.open);
                            self.open[fd] = Some(o);
                            return fd as i64;
                        }
                        if flags & O_CREATE == 0 {
                            return -ENOENT;
                        }
                        let d = Arc::new(Mutex::new(Vec::new()));
                        self.files.insert(path, d.clone());
                        d
                    }
                };
                let o = MemOpen {
                    data,
                    pos: 0,
                    readable: flags & O_READ != 0,
                    writable: flags & (O_WRITE | O_APPEND) != 0,
                    append: flags & O_APPEND != 0,
                };
                let fd = alloc_fd(&mut self.open);
                self.open[fd] = Some(o);
                fd as i64
            }
            FS_READ => {
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                if !o.readable {
                    return -EBADF;
                }
                let (buf, len) = (a(1), a(2));
                if buf < 0 || len < 0 {
                    return -EINVAL;
                }
                let data = o.data.lock().unwrap_or_else(|e| e.into_inner());
                let avail = data.len().saturating_sub(o.pos);
                let n = avail.min(len as usize);
                if n > 0 {
                    let Some(m) = mem.as_deref_mut() else {
                        return -EFAULT;
                    };
                    if m.write_bytes(buf as u64, &data[o.pos..o.pos + n]).is_none() {
                        return -EFAULT;
                    }
                }
                drop(data);
                o.pos += n;
                n as i64
            }
            FS_WRITE => {
                if self.crash_frozen() {
                    return a(2).max(0); // power-loss: the un-synced write is silently dropped
                }
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                if !o.writable {
                    return -EBADF;
                }
                let (buf, len) = (a(1), a(2));
                if buf < 0 || len < 0 {
                    return -EINVAL;
                }
                let bytes = match mem.as_deref() {
                    Some(m) => match m.read_bytes(buf as u64, len as u64) {
                        Some(b) => b,
                        None => return -EFAULT,
                    },
                    None => return -EFAULT,
                };
                let mut data = o.data.lock().unwrap_or_else(|e| e.into_inner());
                if o.append {
                    o.pos = data.len();
                }
                if o.pos > data.len() {
                    data.resize(o.pos, 0); // POSIX: writing past EOF zero-fills the gap
                }
                let end = o.pos + bytes.len();
                if end > data.len() {
                    data.resize(end, 0);
                }
                data[o.pos..end].copy_from_slice(&bytes);
                drop(data);
                o.pos = end;
                bytes.len() as i64
            }
            FS_SEEK => {
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                let size = o.data.lock().unwrap_or_else(|e| e.into_inner()).len() as i64;
                let base = match a(1) {
                    0 => 0,
                    1 => o.pos as i64,
                    2 => size,
                    _ => return -EINVAL,
                };
                let Some(new) = base.checked_add(a(2)) else {
                    return -EINVAL;
                };
                if new < 0 {
                    return -EINVAL;
                }
                o.pos = new as usize;
                new
            }
            FS_CLOSE => {
                let Some(slot) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                if slot.take().is_none() {
                    return -EBADF;
                }
                0
            }
            FS_REMOVE => {
                let path = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                if self.files.remove(&path).is_none() {
                    return -ENOENT;
                }
                0
            }
            FS_RENAME => {
                let from = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let to = match read_path(mem.as_deref(), a(2), a(3)).map(|p| norm(&p)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let Some(d) = self.files.remove(&from) else {
                    return -ENOENT;
                };
                self.files.insert(to, d);
                0
            }
            FS_TRUNCATE => {
                if self.crash_frozen() {
                    return 0; // power-loss: the resize never reaches the backing file
                }
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                if !o.writable {
                    return -EBADF; // POSIX ftruncate needs a writable descriptor
                }
                let len = a(1);
                if len < 0 {
                    return -EINVAL;
                }
                // POSIX: shrink discards, grow zero-fills; the cursor is untouched.
                o.data
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .resize(len as usize, 0);
                0
            }
            FS_SYNC => {
                // Memory is always "durable" here — a durability barrier for crash injection, then
                // validate the fd (nothing to flush).
                if self.crash_barrier() {
                    return 0;
                }
                let Some(Some(_)) = self.open.get(a(0) as usize) else {
                    return -EBADF;
                };
                0
            }
            FS_MMAP => {
                let (fd, foff, len, buf) = (a(0), a(1), a(2), a(3));
                if foff < 0 || len < 0 || buf < 0 {
                    return -EINVAL;
                }
                let Some(Some(o)) = self.open.get(fd as usize) else {
                    return -EBADF;
                };
                let data = o.data.clone();
                // Copy the file region into the guest buffer (zero-fill past EOF, like a file-backed
                // mmap of a hole).
                let mut region = vec![0u8; len as usize];
                {
                    let d = data.lock().unwrap_or_else(|e| e.into_inner());
                    let start = (foff as usize).min(d.len());
                    let end = (foff as usize + len as usize).min(d.len());
                    if end > start {
                        region[..end - start].copy_from_slice(&d[start..end]);
                    }
                }
                let Some(m) = mem.as_deref_mut() else {
                    return -EFAULT;
                };
                if m.write_bytes(buf as u64, &region).is_none() {
                    return -EFAULT;
                }
                self.maps.push(MemMapping {
                    base: buf as u64,
                    len: len as u64,
                    data,
                    file_off: foff as u64,
                });
                0
            }
            FS_MSYNC => {
                let (buf, len) = (a(0), a(1));
                if buf < 0 || len < 0 {
                    return -EINVAL;
                }
                let Some(map) = self
                    .maps
                    .iter()
                    .find(|m| buf as u64 >= m.base && (buf as u64) < m.base + m.len)
                else {
                    return -EINVAL; // no mapping contains this address
                };
                let n = (len as u64).min(map.base + map.len - buf as u64) as usize;
                let file_pos = map.file_off + (buf as u64 - map.base);
                let data = map.data.clone(); // end the borrow of `self.maps` before `crash_barrier`
                let Some(m) = mem.as_deref() else {
                    return -EFAULT;
                };
                let Some(bytes) = m.read_bytes(buf as u64, n as u64) else {
                    return -EFAULT;
                };
                if self.crash_barrier() {
                    return 0; // power-loss: this msync's bytes never reach the file
                }
                let mut d = data.lock().unwrap_or_else(|e| e.into_inner());
                let end = file_pos as usize + n;
                if end > d.len() {
                    d.resize(end, 0);
                }
                d[file_pos as usize..end].copy_from_slice(&bytes);
                0
            }
            FS_MUNMAP => {
                let buf = a(0);
                // Flush the whole mapping, then drop it (LMDB msyncs explicitly before close, but a
                // final flush keeps `munmap` self-contained) — unless a crash has frozen the store,
                // in which case a real `munmap` on a dead process would flush nothing.
                let Some(idx) = self.maps.iter().position(|m| m.base == buf as u64) else {
                    return -EINVAL;
                };
                let map = self.maps.remove(idx);
                if !self.crash_frozen() {
                    if let Some(m) = mem.as_deref() {
                        if let Some(bytes) = m.read_bytes(map.base, map.len) {
                            let mut d = map.data.lock().unwrap_or_else(|e| e.into_inner());
                            let end = (map.file_off + map.len) as usize;
                            if end > d.len() {
                                d.resize(end, 0);
                            }
                            d[map.file_off as usize..end].copy_from_slice(&bytes);
                        }
                    }
                }
                0
            }
            FS_CRASH_ARM => arm_crash(self.crash.as_mut(), a(0)),
            FS_STAT => {
                let key = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(k) => k,
                    Err(e) => return e,
                };
                // A file name wins over a same-named implicit dir prefix (there can't be both).
                let (mode, size) = if let Some(d) = self.files.get(&key) {
                    let len = d.lock().unwrap_or_else(|e| e.into_inner()).len() as i64;
                    (S_IFREG | 0o644, len)
                } else if self.is_dir(&key) {
                    // Owner-only (0700), the natural mode for a hermetic in-memory store — and the mode
                    // Postgres' `checkDataDir` demands of its data directory (0700 or 0750; a
                    // world/group-readable data dir is a fatal error). Differential stat tests compare
                    // only `S_IFMT`, so the perm bits are free to model an owner-private fs.
                    (S_IFDIR | 0o700, 0)
                } else {
                    return -ENOENT;
                };
                // A stable synthetic identity: mtime 0, one link, a hash-free ino of 0 (Postgres uses
                // st_ino/st_dev only for cross-file identity, which a fresh per-run store never needs).
                let buf = stat_bytes(mode, 1, size, 0, 0, 0, 0, 0, 0, 4096, (size + 511) / 512);
                if a(2) < 0 || a(3) < STATBUF_LEN as i64 {
                    return -EINVAL;
                }
                let Some(m) = mem.as_deref_mut() else {
                    return -EFAULT;
                };
                if m.write_bytes(a(2) as u64, &buf).is_some() {
                    0
                } else {
                    -EFAULT
                }
            }
            FS_MKDIR => {
                let key = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(k) => k,
                    Err(e) => return e,
                };
                if key.is_empty() || self.is_dir(&key) || self.files.contains_key(&key) {
                    return -EEXIST;
                }
                self.dirs.insert(key);
                0
            }
            FS_RMDIR => {
                let key = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(k) => k,
                    Err(e) => return e,
                };
                if self.files.contains_key(&key) {
                    return -ENOTDIR;
                }
                if !self.is_dir(&key) {
                    return -ENOENT;
                }
                if !self.children_of(&key).is_empty() {
                    return -ENOTEMPTY;
                }
                if !self.dirs.remove(&key) {
                    return -ENOENT; // an implicit (non-empty) dir has no explicit entry to remove
                }
                0
            }
            FS_OPENDIR => {
                let key = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(k) => k,
                    Err(e) => return e,
                };
                if self.files.contains_key(&key) {
                    return -ENOTDIR;
                }
                if !self.is_dir(&key) {
                    return -ENOENT;
                }
                let mut kids = self.children_of(&key);
                kids.reverse(); // pop() yields them in sorted order
                let dh = self.opendirs.iter().position(|s| s.is_none()).unwrap_or({
                    self.opendirs.push(None);
                    self.opendirs.len() - 1
                });
                self.opendirs[dh] = Some(kids);
                dh as i64
            }
            FS_READDIR => {
                let Some(Some(entries)) = self.opendirs.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                let Some(name) = entries.pop() else {
                    return 0; // exhausted
                };
                let (ptr, cap) = (a(1), a(2));
                if ptr < 0 || cap < name.len() as i64 {
                    entries.push(name); // leave the walk where it was
                    return -EINVAL;
                }
                let Some(m) = mem else {
                    return -EFAULT;
                };
                if m.write_bytes(ptr as u64, name.as_bytes()).is_some() {
                    name.len() as i64
                } else {
                    -EFAULT
                }
            }
            FS_CLOSEDIR => {
                let Some(slot) = self.opendirs.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                if slot.take().is_none() {
                    return -EBADF;
                }
                0
            }
            _ => -EINVAL,
        }
    }
}

/// [`FS_CRASH_ARM`] handler shared by both backends: `n < 0` disarms, `n >= 0` arms the crash to trip
/// after `n` further durability barriers. `-EINVAL` when the backend has no controller (the default,
/// non-`crashy` grants) — so the op simply does not exist on a shipping capability.
fn arm_crash(ctl: Option<&mut CrashCtl>, n: i64) -> i64 {
    let Some(c) = ctl else {
        return -EINVAL;
    };
    c.countdown = if n < 0 { None } else { Some(n as u64) };
    c.crashed = false;
    0
}

/// A deterministic **in-memory** filesystem capability (fresh, empty state per host). The hermetic
/// default for tests and differential runs.
pub fn mem_fs() -> HostCap {
    mem_fs_impl(false)
}

/// Like [`mem_fs`] but with the **test-only crash-injection** controller enabled (the [`FS_CRASH_ARM`]
/// op becomes live). Used to prove crash-consistency of a mapped store — see `demo_lmdb_crash_recovery`.
/// Never grant this to a real guest: a tripped crash freezes the store (a self-inflicted DoS on the
/// holder's own fs, no host effect, but pointless outside a test).
pub fn mem_fs_crashy() -> HostCap {
    mem_fs_impl(true)
}

fn mem_fs_impl(crashy: bool) -> HostCap {
    HostCap::host_fn(0, move || {
        let mut st = MemFsState {
            crash: crashy.then(CrashCtl::default),
            ..MemFsState::default()
        };
        Box::new(
            move |op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>| {
                Ok(vec![st.handle(op, args, mem)])
            },
        ) as HostFn
    })
}

/// A **pre-seeded** in-memory filesystem: `files` maps a normalized relative path to its contents, and
/// `dirs` names directories that must exist even with no files under them (empty dirs a walk would
/// otherwise miss). Each host grant gets a **fresh clone** of the seed, so a run's writes never leak
/// back into it — re-runs are deterministic. This is how a demo with no real filesystem (e.g. the
/// browser) mounts a data-dir image on the `fs` cap: build the seed once (`mem_fs_from_host_dir` or a
/// shipped image), then grant it. Confinement is unchanged — the same rooted, `..`-refusing store as
/// [`mem_fs`], just non-empty at grant time.
pub fn mem_fs_seeded(files: Vec<(String, Vec<u8>)>, dirs: Vec<String>) -> HostCap {
    let files = Arc::new(files);
    let dirs = Arc::new(dirs);
    HostCap::host_fn(0, move || {
        let (files, dirs) = (files.clone(), dirs.clone());
        let mut st = MemFsState::default();
        for (p, data) in files.iter() {
            st.files.insert(norm(p), Arc::new(Mutex::new(data.clone())));
        }
        for d in dirs.iter() {
            st.dirs.insert(norm(d));
        }
        Box::new(
            move |op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>| {
                Ok(vec![st.handle(op, args, mem)])
            },
        ) as HostFn
    })
}

/// Walk a host directory into a [`mem_fs_seeded`] capability — every regular file's bytes keyed by its
/// path relative to `root`, plus every directory (so empty ones survive). Symlinks are followed (a
/// symlink to a file becomes that file's bytes). For building a demo data image from an on-disk
/// cluster; the resulting cap has no further tie to the host filesystem.
pub fn mem_fs_from_host_dir(root: &Path) -> std::io::Result<HostCap> {
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut dirs: Vec<String> = Vec::new();
    fn walk(
        base: &Path,
        cur: &Path,
        files: &mut Vec<(String, Vec<u8>)>,
        dirs: &mut Vec<String>,
    ) -> std::io::Result<()> {
        for entry in std::fs::read_dir(cur)? {
            let path = entry?.path();
            let rel = path
                .strip_prefix(base)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            // `is_dir`/`is_file` follow symlinks, so a symlinked file/dir is captured as its target.
            if path.is_dir() {
                dirs.push(rel);
                walk(base, &path, files, dirs)?;
            } else if path.is_file() {
                files.push((rel, std::fs::read(&path)?));
            }
        }
        Ok(())
    }
    walk(root, root, &mut files, &mut dirs)?;
    Ok(mem_fs_seeded(files, dirs))
}

struct HostOpen {
    file: std::fs::File,
    readable: bool,
    writable: bool,
}

/// A live `mmap` over the real fs: the guest buffer `[base, base+len)` is bound to the open file at
/// `open_idx`, starting at `file_off`. `msync` `pwrite`s a sub-range back through that file.
struct HostMapping {
    base: u64,
    len: u64,
    open_idx: usize,
    file_off: u64,
}

struct HostFsState {
    root: PathBuf,
    open: Vec<Option<HostOpen>>,
    /// Snapshots taken by [`FS_OPENDIR`]: `opendirs[dh]` is the remaining child names to yield
    /// (sorted, so a walk is deterministic and matches `mem_fs`).
    opendirs: Vec<Option<Vec<String>>>,
    maps: Vec<HostMapping>,
    /// `Some` only on the `host_fs_crashy` variant (test-only crash injection); `None` on `host_fs`.
    crash: Option<CrashCtl>,
}

impl HostFsState {
    /// A durability barrier (`msync`/`sync`): `true` ⇒ drop this write (crashed or crashing now).
    fn crash_barrier(&mut self) -> bool {
        self.crash.as_mut().is_some_and(CrashCtl::barrier)
    }
    /// Whether the backing store is frozen by a tripped crash (persisting ops become no-ops).
    fn crash_frozen(&self) -> bool {
        self.crash.as_ref().is_some_and(|c| c.crashed)
    }

    fn handle(
        &mut self,
        op: u32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
        minter: Option<&mut dyn RegionMinter>,
    ) -> i64 {
        let mut mem = mem;
        let a = |i: usize| args.get(i).copied().unwrap_or(0);
        // `read_path` already refused anything that could escape (absolute / `..` / empty), so the
        // join below cannot leave `root` — the rooted directory *is* the attenuation.
        let path_at =
            |mem: Option<&dyn GuestMem>, root: &Path, p: i64, l: i64| -> Result<PathBuf, i64> {
                Ok(root.join(read_path(mem, p, l)?))
            };
        match op {
            FS_OPEN => {
                let path = match path_at(mem.as_deref(), &self.root, a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let flags = a(2);
                let mut oo = std::fs::OpenOptions::new();
                oo.read(flags & O_READ != 0);
                if flags & O_APPEND != 0 {
                    oo.append(true);
                } else {
                    oo.write(flags & O_WRITE != 0);
                }
                oo.truncate(flags & O_TRUNC != 0)
                    .create(flags & O_CREATE != 0);
                let file = match oo.open(&path) {
                    Ok(f) => f,
                    Err(e) => return -io_errno(&e),
                };
                let o = HostOpen {
                    file,
                    readable: flags & O_READ != 0,
                    writable: flags & (O_WRITE | O_APPEND) != 0,
                };
                let fd = alloc_fd(&mut self.open);
                self.open[fd] = Some(o);
                fd as i64
            }
            FS_READ => {
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                if !o.readable {
                    return -EBADF;
                }
                let (buf, len) = (a(1), a(2));
                if buf < 0 || !(0..=(1 << 30)).contains(&len) {
                    return -EINVAL;
                }
                let mut tmp = vec![0u8; len as usize];
                let n = match o.file.read(&mut tmp) {
                    Ok(n) => n,
                    Err(e) => return -io_errno(&e),
                };
                if n > 0 {
                    let Some(m) = mem.as_deref_mut() else {
                        return -EFAULT;
                    };
                    if m.write_bytes(buf as u64, &tmp[..n]).is_none() {
                        return -EFAULT;
                    }
                }
                n as i64
            }
            FS_WRITE => {
                if self.crash_frozen() {
                    return a(2).max(0); // power-loss: the un-synced write is silently dropped
                }
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                if !o.writable {
                    return -EBADF;
                }
                let bytes = match mem.as_deref() {
                    Some(m) => match m.read_bytes(a(1) as u64, a(2) as u64) {
                        Some(b) => b,
                        None => return -EFAULT,
                    },
                    None => return -EFAULT,
                };
                match o.file.write_all(&bytes) {
                    Ok(()) => bytes.len() as i64,
                    Err(e) => -io_errno(&e),
                }
            }
            FS_SEEK => {
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                let from = match a(1) {
                    0 => SeekFrom::Start(a(2).max(0) as u64),
                    1 => SeekFrom::Current(a(2)),
                    2 => SeekFrom::End(a(2)),
                    _ => return -EINVAL,
                };
                match o.file.seek(from) {
                    Ok(p) => p as i64,
                    Err(e) => -io_errno(&e),
                }
            }
            FS_CLOSE => {
                let Some(slot) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                if slot.take().is_none() {
                    return -EBADF;
                }
                0
            }
            FS_REMOVE => {
                let path = match path_at(mem.as_deref(), &self.root, a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                match std::fs::remove_file(path) {
                    Ok(()) => 0,
                    Err(e) => -io_errno(&e),
                }
            }
            FS_RENAME => {
                let from = match path_at(mem.as_deref(), &self.root, a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let to = match path_at(mem.as_deref(), &self.root, a(2), a(3)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                match std::fs::rename(from, to) {
                    Ok(()) => 0,
                    Err(e) => -io_errno(&e),
                }
            }
            FS_TRUNCATE => {
                if self.crash_frozen() {
                    return 0; // power-loss: the resize never reaches the backing file
                }
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                if !o.writable {
                    return -EBADF;
                }
                let len = a(1);
                if len < 0 {
                    return -EINVAL;
                }
                match o.file.set_len(len as u64) {
                    Ok(()) => 0,
                    Err(e) => -io_errno(&e),
                }
            }
            FS_SYNC => {
                // A durability barrier for crash injection, then the real fsync.
                if self.crash_barrier() {
                    return 0;
                }
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                match o.file.sync_all() {
                    Ok(()) => 0,
                    Err(e) => -io_errno(&e),
                }
            }
            FS_MMAP => {
                let (fd, foff, len, buf) = (a(0), a(1), a(2), a(3));
                if foff < 0 || len < 0 || buf < 0 {
                    return -EINVAL;
                }
                let Some(Some(o)) = self.open.get_mut(fd as usize) else {
                    return -EBADF;
                };
                // Copy the file region into the guest buffer (zero-fill past EOF).
                let mut region = vec![0u8; len as usize];
                if o.file.seek(SeekFrom::Start(foff as u64)).is_err() {
                    return -io_errno(&std::io::Error::last_os_error());
                }
                let mut got = 0usize;
                while got < region.len() {
                    match o.file.read(&mut region[got..]) {
                        Ok(0) => break, // EOF — rest stays zero
                        Ok(n) => got += n,
                        Err(e) => return -io_errno(&e),
                    }
                }
                let Some(m) = mem.as_deref_mut() else {
                    return -EFAULT;
                };
                if m.write_bytes(buf as u64, &region).is_none() {
                    return -EFAULT;
                }
                self.maps.push(HostMapping {
                    base: buf as u64,
                    len: len as u64,
                    open_idx: fd as usize,
                    file_off: foff as u64,
                });
                0
            }
            FS_MSYNC => {
                let (buf, len) = (a(0), a(1));
                if buf < 0 || len < 0 {
                    return -EINVAL;
                }
                let Some(map) = self
                    .maps
                    .iter()
                    .find(|m| buf as u64 >= m.base && (buf as u64) < m.base + m.len)
                else {
                    return -EINVAL;
                };
                let n = (len as u64).min(map.base + map.len - buf as u64) as usize;
                let file_pos = map.file_off + (buf as u64 - map.base);
                let open_idx = map.open_idx;
                let Some(m) = mem.as_deref() else {
                    return -EFAULT;
                };
                let Some(bytes) = m.read_bytes(buf as u64, n as u64) else {
                    return -EFAULT;
                };
                if self.crash_barrier() {
                    return 0; // power-loss: this msync's bytes never reach the file
                }
                let Some(Some(o)) = self.open.get_mut(open_idx) else {
                    return -EBADF; // the mapped fd was closed
                };
                if o.file.seek(SeekFrom::Start(file_pos)).is_err() {
                    return -io_errno(&std::io::Error::last_os_error());
                }
                match o.file.write_all(&bytes) {
                    Ok(()) => 0,
                    Err(e) => -io_errno(&e),
                }
            }
            FS_MUNMAP => {
                let buf = a(0);
                let Some(idx) = self.maps.iter().position(|m| m.base == buf as u64) else {
                    return -EINVAL;
                };
                let map = self.maps.remove(idx);
                // Final flush of the whole mapping (self-contained munmap) — unless a crash froze the
                // store, in which case a real munmap on a dead process would flush nothing.
                if !self.crash_frozen() {
                    let bytes = mem.as_deref().and_then(|m| m.read_bytes(map.base, map.len));
                    if let (Some(bytes), Some(Some(o))) = (bytes, self.open.get_mut(map.open_idx)) {
                        let _ = o.file.seek(SeekFrom::Start(map.file_off));
                        let _ = o.file.write_all(&bytes);
                    }
                }
                0
            }
            FS_CRASH_ARM => arm_crash(self.crash.as_mut(), a(0)),
            // §4b zero-copy: mint a file-backed `SharedRegion` over the open file and return its
            // handle for the guest to `SharedRegion.map` into its window (real MAP_SHARED aliasing).
            FS_MAP_REGION => {
                let Some(minter) = minter else {
                    return -EINVAL; // not an mmap-capable grant (plain host_fs / mem_fs)
                };
                let (fd, foff, len) = (a(0), a(1), a(2));
                if foff != 0 || len <= 0 {
                    return -EINVAL; // v1 maps the whole file from offset 0
                }
                let Some(Some(o)) = self.open.get(fd as usize) else {
                    return -EBADF;
                };
                // Give the backing its **own** fd (dup) over the same OS file, so it outlives the
                // guest's fd and both share one page cache — the map and the fs cap's pread/pwrite/
                // fsync stay coherent. Unix only for now (the `FileBacking` is `#[cfg(unix)]`);
                // elsewhere the op is unavailable and the guest shim falls back to copy-in.
                #[cfg(unix)]
                {
                    let dup = match o.file.try_clone() {
                        Ok(f) => f,
                        Err(e) => return -io_errno(&e),
                    };
                    match crate::new_file_region(dup, len as usize) {
                        Ok(backing) => minter.grant_region(backing) as i64,
                        Err(e) => -io_errno(&e),
                    }
                }
                #[cfg(not(unix))]
                {
                    let _ = (minter, o, len); // region-mapping is unix-only for now
                    -EINVAL
                }
            }
            FS_STAT => {
                let path = match path_at(mem.as_deref(), &self.root, a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                // lstat semantics (symlink_metadata): a symlink is reported as a symlink and never
                // followed, so its type can't be confused with a target outside the granted root.
                let md = match std::fs::symlink_metadata(&path) {
                    Ok(m) => m,
                    Err(e) => return -io_errno(&e),
                };
                let buf = host_stat_bytes(&md);
                if a(2) < 0 || a(3) < STATBUF_LEN as i64 {
                    return -EINVAL;
                }
                let Some(m) = mem.as_deref_mut() else {
                    return -EFAULT;
                };
                if m.write_bytes(a(2) as u64, &buf).is_some() {
                    0
                } else {
                    -EFAULT
                }
            }
            FS_MKDIR => {
                let path = match path_at(mem.as_deref(), &self.root, a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                match std::fs::create_dir(path) {
                    Ok(()) => 0,
                    Err(e) => -io_errno(&e),
                }
            }
            FS_RMDIR => {
                let path = match path_at(mem.as_deref(), &self.root, a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                match std::fs::remove_dir(path) {
                    Ok(()) => 0,
                    Err(e) => -io_errno(&e),
                }
            }
            FS_OPENDIR => {
                let path = match path_at(mem.as_deref(), &self.root, a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let rd = match std::fs::read_dir(&path) {
                    Ok(r) => r,
                    Err(e) => return -io_errno(&e),
                };
                let mut names = Vec::new();
                for ent in rd {
                    match ent {
                        Ok(e) => {
                            // Names are OS strings; a non-UTF-8 entry can't round-trip through the
                            // guest's UTF-8 path protocol, so skip it (it was never nameable anyway).
                            if let Some(s) = e.file_name().to_str() {
                                names.push(s.to_string());
                            }
                        }
                        Err(e) => return -io_errno(&e),
                    }
                }
                names.sort();
                names.reverse(); // pop() yields sorted order
                let dh = self.opendirs.iter().position(|s| s.is_none()).unwrap_or({
                    self.opendirs.push(None);
                    self.opendirs.len() - 1
                });
                self.opendirs[dh] = Some(names);
                dh as i64
            }
            FS_READDIR => {
                let Some(Some(entries)) = self.opendirs.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                let Some(name) = entries.pop() else {
                    return 0; // exhausted
                };
                let (ptr, cap) = (a(1), a(2));
                if ptr < 0 || cap < name.len() as i64 {
                    entries.push(name);
                    return -EINVAL;
                }
                let Some(m) = mem else {
                    return -EFAULT;
                };
                if m.write_bytes(ptr as u64, name.as_bytes()).is_some() {
                    name.len() as i64
                } else {
                    -EFAULT
                }
            }
            FS_CLOSEDIR => {
                let Some(slot) = self.opendirs.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                if slot.take().is_none() {
                    return -EBADF;
                }
                0
            }
            _ => -EINVAL,
        }
    }
}

/// Fill a [`StatBuf`] from real-fs metadata. On unix the `st_*` fields come straight from the
/// `stat(2)` the OS already did; elsewhere the portable `Metadata` fills type + size and the rest
/// stays zero (the guest libc only ever needs type + size + mtime to walk an `initdb` tree).
fn host_stat_bytes(md: &std::fs::Metadata) -> [u8; STATBUF_LEN] {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        stat_bytes(
            md.mode(),
            md.nlink() as u32,
            md.size() as i64,
            md.mtime(),
            md.mtime_nsec(),
            md.ino(),
            md.dev(),
            md.uid(),
            md.gid(),
            md.blksize() as i64,
            md.blocks() as i64,
        )
    }
    #[cfg(not(unix))]
    {
        let ty = if md.is_dir() {
            S_IFDIR | 0o755
        } else if md.file_type().is_symlink() {
            S_IFLNK | 0o777
        } else {
            S_IFREG | 0o644
        };
        stat_bytes(ty, 1, md.len() as i64, 0, 0, 0, 0, 0, 0, 4096, 0)
    }
}

/// Map a host `io::Error` to the protocol's canonical (Linux) errno. The wire protocol must be
/// **host-OS-independent** — both backends have to return the same value for the same failure, and
/// on a differential a `host_fs` run has to match a `mem_fs` run — but the raw OS error code is not
/// portable: `ENOTEMPTY` is 39 on Linux and 66 on macOS, and Windows codes are not errno at all. So
/// canonicalize through the portable `io::ErrorKind`, falling back to the raw errno (already the
/// canonical value on Linux, the differential's reference host) for kinds without a fixed mapping.
fn io_errno(e: &std::io::Error) -> i64 {
    use std::io::ErrorKind;
    match e.kind() {
        ErrorKind::NotFound => ENOENT,
        ErrorKind::PermissionDenied => EACCES,
        ErrorKind::AlreadyExists => EEXIST,
        ErrorKind::DirectoryNotEmpty => ENOTEMPTY,
        ErrorKind::NotADirectory => ENOTDIR,
        ErrorKind::InvalidInput => EINVAL,
        _ => e.raw_os_error().map(|c| c as i64).unwrap_or(EINVAL),
    }
}

/// The **real** filesystem, attenuated to `root` (relative paths only; `..`/absolute refused). The
/// guest sees exactly the subtree the embedder granted — nothing else is nameable.
pub fn host_fs(root: PathBuf) -> HostCap {
    host_fs_impl(root, false)
}

/// Like [`host_fs`] but with the **test-only crash-injection** controller enabled (the
/// [`FS_CRASH_ARM`] op becomes live). Proves crash-consistency of a real on-disk mapped store — see
/// `demo_lmdb_crash_recovery`. Never grant this to a real guest.
pub fn host_fs_crashy(root: PathBuf) -> HostCap {
    host_fs_impl(root, true)
}

/// Like [`host_fs`] but **mmap-capable** (§4b): granted with region-minting authority so the
/// [`FS_MAP_REGION`] op is live. A guest that maps a file gets a **real** `MAP_SHARED` alias of that
/// file into its window (zero-copy, page-cache coherent with the same file's `pread`/`pwrite`) instead
/// of the copy-in emulation. The guest shim falls back to `FS_MMAP` when granted a plain `host_fs`.
pub fn host_fs_mmap(root: PathBuf) -> HostCap {
    HostCap::host_fn_region(0, move || {
        let mut st = HostFsState {
            root: root.clone(),
            open: Vec::new(),
            opendirs: Vec::new(),
            maps: Vec::new(),
            crash: None,
        };
        Box::new(
            move |op: u32,
                  args: &[i64],
                  mem: Option<&mut dyn GuestMem>,
                  minter: &mut dyn RegionMinter| {
                Ok(vec![st.handle(op, args, mem, Some(minter))])
            },
        ) as HostFnRegion
    })
}

fn host_fs_impl(root: PathBuf, crashy: bool) -> HostCap {
    HostCap::host_fn(0, move || {
        let mut st = HostFsState {
            root: root.clone(),
            open: Vec::new(),
            opendirs: Vec::new(),
            maps: Vec::new(),
            crash: crashy.then(CrashCtl::default),
        };
        Box::new(
            move |op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>| {
                Ok(vec![st.handle(op, args, mem, None)])
            },
        ) as HostFn
    })
}

#[cfg(test)]
mod tests {
    /// `svm-llvm` pins the `HostFn` interface id numerically (it produces `svm-ir` and cannot depend
    /// on the interpreter crate); this locks the pin to the real constant.
    #[test]
    fn host_fn_type_id_matches() {
        assert_eq!(svm_interp::iface::HOST_FN, 13);
    }

    /// `svm-llvm` pins `SharedRegion`'s interface id numerically (`SHARED_REGION_TYPE_ID`, for the
    /// `__vm_region_call` the zero-copy bridge lowers to); this locks that pin to the real constant.
    #[test]
    fn shared_region_type_id_matches() {
        assert_eq!(svm_interp::iface::SHARED_REGION, 4);
    }

    /// The §4b zero-copy op end to end at the capability boundary: `host_fs_mmap`'s `FS_MAP_REGION`
    /// opens a real file and mints a **file-backed `SharedRegion`** whose size matches, while a plain
    /// `host_fs` refuses the op (no minting authority). Combined with `file_region_tests` (which prove
    /// a `FileBacking` aliases the real file's bytes), this establishes the whole delivery path:
    /// `FS_MAP_REGION` → `FileBacking` → a live `SharedRegion` the guest can `SharedRegion.map`.
    #[cfg(unix)]
    #[test]
    fn map_region_op_mints_a_file_backed_region_only_on_the_mmap_grant() {
        use super::*;
        use std::io::Write;
        use svm_interp::{iface, Host, WindowMem};

        let dir = std::env::temp_dir().join(format!("svm_fs_mapregion_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        {
            let mut f = std::fs::File::create(dir.join("data")).unwrap();
            f.write_all(b"ZEROCOPY").unwrap();
            f.set_len(8192).unwrap();
        }

        // A tiny flat window holding the path "data" at offset 0.
        let mut win = vec![0u8; 4096];
        win[..4].copy_from_slice(b"data");
        let open_args = [0i64, 4, O_READ | O_WRITE, 0];

        // 1. The mmap-capable grant: open → map_region mints a live region of the file's size.
        {
            let mut host = Host::new();
            let cap = host_fs_mmap(dir.clone());
            let fs_h = (cap.grant)(&mut host, 0);
            let mut wm = WindowMem::new(&mut win, 4096);
            let fd = host
                .cap_dispatch_slots(iface::HOST_FN, FS_OPEN, fs_h, &open_args, Some(&mut wm))
                .unwrap()[0];
            assert!(fd >= 0, "open failed: {fd}");
            let region = host
                .cap_dispatch_slots(
                    iface::HOST_FN,
                    FS_MAP_REGION,
                    fs_h,
                    &[fd, 0, 8192],
                    Some(&mut wm),
                )
                .unwrap()[0];
            assert!(
                region >= 0,
                "FS_MAP_REGION should mint a region on host_fs_mmap: {region}"
            );
            // The returned handle is a live SharedRegion; op 2 `len()` reports the mapped size.
            let len = host
                .cap_dispatch_slots(iface::SHARED_REGION, 2, region as i32, &[], Some(&mut wm))
                .unwrap()[0];
            assert_eq!(len, 8192, "the minted region maps the whole file");
        }

        // 2. A plain host_fs has no minting authority → the op is unavailable (-EINVAL).
        {
            let mut host = Host::new();
            let cap = host_fs(dir.clone());
            let fs_h = (cap.grant)(&mut host, 0);
            let mut wm = WindowMem::new(&mut win, 4096);
            let fd = host
                .cap_dispatch_slots(iface::HOST_FN, FS_OPEN, fs_h, &open_args, Some(&mut wm))
                .unwrap()[0];
            let region = host
                .cap_dispatch_slots(
                    iface::HOST_FN,
                    FS_MAP_REGION,
                    fs_h,
                    &[fd, 0, 8192],
                    Some(&mut wm),
                )
                .unwrap()[0];
            assert_eq!(region, -EINVAL, "plain host_fs must refuse FS_MAP_REGION");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The metadata + directory surface (`stat`/`mkdir`/`rmdir`/`opendir`/`readdir`/`closedir`) is
    /// protocol-identical on both backends: the *same* scripted walk returns the *same* rc sequence,
    /// the same file/dir type bits, and the same size on `mem_fs` (flat-map dirs) and `host_fs` (real
    /// dirs). This is the invariant a Postgres differential relies on — the guest can't tell which
    /// backend it walks.
    #[test]
    fn os_metadata_ops_parity_mem_vs_host() {
        use super::*;
        use svm_interp::{iface, Host, WindowMem};

        // Window layout: [0..64) path A, [64..128) path B, [256..328) statbuf, [512..576) name buf.
        const PA: usize = 0;
        const PB: usize = 64;
        const SB: usize = 256;
        const NB: usize = 512;

        // Drive one op over a fresh window view; `win` persists between calls (paths stay put).
        fn call(host: &mut Host, h: i32, win: &mut [u8], op: u32, args: &[i64]) -> i64 {
            let n = win.len();
            let mut wm = WindowMem::new(win, n as u64);
            host.cap_dispatch_slots(iface::HOST_FN, op, h, args, Some(&mut wm))
                .unwrap()[0]
        }
        fn put(win: &mut [u8], at: usize, s: &str) -> (i64, i64) {
            win[at..at + s.len()].copy_from_slice(s.as_bytes());
            (at as i64, s.len() as i64)
        }

        // The scripted walk: returns (rc-sequence, stat-of-dir mode&S_IFMT, stat-of-file (mode,size),
        // readdir name). Run identically against whatever cap is granted.
        fn walk(host: &mut Host, h: i32, win: &mut [u8]) -> (Vec<i64>, u32, (u32, i64), String) {
            let mut rc = Vec::new();
            let (dp, dl) = put(win, PA, "d");
            rc.push(call(host, h, win, FS_MKDIR, &[dp, dl, 0, 0])); // 0
            rc.push(call(host, h, win, FS_MKDIR, &[dp, dl, 0, 0])); // -EEXIST
            rc.push(call(
                host,
                h,
                win,
                FS_STAT,
                &[dp, dl, SB as i64, STATBUF_LEN as i64],
            )); // 0
            let dir_mode = u32::from_le_bytes(win[SB..SB + 4].try_into().unwrap()) & S_IFMT;

            let (fp, fl) = put(win, PB, "d/f");
            let fd = call(host, h, win, FS_OPEN, &[fp, fl, O_CREATE | O_WRITE, 0]);
            rc.push((fd >= 0) as i64); // 1 (fd is backend-specific; normalize to a bool)
            let (bp, bl) = put(win, NB, "hello");
            rc.push(call(host, h, win, FS_WRITE, &[fd, bp, bl, 0])); // 5
            rc.push(call(host, h, win, FS_CLOSE, &[fd, 0, 0, 0])); // 0

            // re-put the path (the write clobbered NB, not PB, but keep it explicit)
            let (fp, fl) = put(win, PB, "d/f");
            rc.push(call(
                host,
                h,
                win,
                FS_STAT,
                &[fp, fl, SB as i64, STATBUF_LEN as i64],
            )); // 0
            let f_mode = u32::from_le_bytes(win[SB..SB + 4].try_into().unwrap()) & S_IFMT;
            let f_size = i64::from_le_bytes(win[SB + 8..SB + 16].try_into().unwrap());

            let (np, nl) = put(win, PA, "d/nope");
            rc.push(call(
                host,
                h,
                win,
                FS_STAT,
                &[np, nl, SB as i64, STATBUF_LEN as i64],
            )); // -ENOENT

            let (dp, dl) = put(win, PA, "d");
            let dh = call(host, h, win, FS_OPENDIR, &[dp, dl, 0, 0]);
            rc.push((dh >= 0) as i64); // 1
            let got = call(host, h, win, FS_READDIR, &[dh, NB as i64, 64, 0]);
            rc.push(got); // 1 (len of "f")
            let name = String::from_utf8(win[NB..NB + got.max(0) as usize].to_vec()).unwrap();
            rc.push(call(host, h, win, FS_READDIR, &[dh, NB as i64, 64, 0])); // 0 (exhausted)
            rc.push(call(host, h, win, FS_CLOSEDIR, &[dh, 0, 0, 0])); // 0

            rc.push(call(host, h, win, FS_RMDIR, &[dp, dl, 0, 0])); // -ENOTEMPTY
            let (fp, fl) = put(win, PB, "d/f");
            rc.push(call(host, h, win, FS_REMOVE, &[fp, fl, 0, 0])); // 0
            rc.push(call(host, h, win, FS_RMDIR, &[dp, dl, 0, 0])); // 0
            rc.push(call(
                host,
                h,
                win,
                FS_STAT,
                &[dp, dl, SB as i64, STATBUF_LEN as i64],
            )); // -ENOENT
            (rc, dir_mode, (f_mode, f_size), name)
        }

        // mem_fs
        let (mem_rc, mem_dir, mem_file, mem_name) = {
            let mut host = Host::new();
            let cap = mem_fs();
            let h = (cap.grant)(&mut host, 0);
            let mut win = vec![0u8; 4096];
            walk(&mut host, h, &mut win)
        };

        // host_fs over a fresh temp root
        let root = std::env::temp_dir().join(format!("svm_fs_osmeta_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let (host_rc, host_dir, host_file, host_name) = {
            let mut host = Host::new();
            let cap = host_fs(root.clone());
            let h = (cap.grant)(&mut host, 0);
            let mut win = vec![0u8; 4096];
            walk(&mut host, h, &mut win)
        };
        let _ = std::fs::remove_dir_all(&root);

        let expected: Vec<i64> = vec![
            0, -EEXIST, 0, 1, 5, 0, 0, -ENOENT, 1, 1, 0, 0, -ENOTEMPTY, 0, 0, -ENOENT,
        ];
        assert_eq!(mem_rc, expected, "mem_fs rc sequence");
        assert_eq!(host_rc, expected, "host_fs rc sequence (must match mem_fs)");
        assert_eq!(mem_dir, S_IFDIR, "mem_fs: 'd' is a directory");
        assert_eq!(host_dir, S_IFDIR, "host_fs: 'd' is a directory");
        assert_eq!(
            mem_file,
            (S_IFREG, 5),
            "mem_fs: 'd/f' is a 5-byte regular file"
        );
        assert_eq!(
            host_file,
            (S_IFREG, 5),
            "host_fs: 'd/f' is a 5-byte regular file"
        );
        assert_eq!(mem_name, "f");
        assert_eq!(host_name, "f");
    }

    /// A **pre-seeded** in-memory fs ([`mem_fs_seeded`]) exposes its files and empty dirs, tolerates a
    /// `./` path prefix (all file ops normalize consistently), reports its data-dir-style `0700` dir
    /// mode, and lets a directory be `open`ed read-only and `sync`ed — the exact surface the Postgres
    /// demo boot needs from a virtual filesystem (`BOOTSPEED.md` Milestone A).
    #[test]
    fn seeded_mem_fs_open_norm_and_dir_fsync() {
        use super::*;
        use svm_interp::{iface, Host, WindowMem};

        const PA: usize = 0;
        const SB: usize = 256;
        const RB: usize = 512;
        fn call(host: &mut Host, h: i32, win: &mut [u8], op: u32, args: &[i64]) -> i64 {
            let n = win.len();
            let mut wm = WindowMem::new(win, n as u64);
            host.cap_dispatch_slots(iface::HOST_FN, op, h, args, Some(&mut wm))
                .unwrap()[0]
        }
        fn put(win: &mut [u8], at: usize, s: &str) -> (i64, i64) {
            win[at..at + s.len()].copy_from_slice(s.as_bytes());
            (at as i64, s.len() as i64)
        }

        // Seed a file under a subdir + an *empty* subdir (the shape of `pg_logical/mappings`).
        let cap = mem_fs_seeded(
            vec![("sub/PG_VERSION".into(), b"17\n".to_vec())],
            vec!["sub/mappings".into()],
        );
        let mut host = Host::new();
        let h = (cap.grant)(&mut host, 0);
        let mut win = vec![0u8; 4096];

        // Open the seeded file through a `./`-prefixed path — the raw key ("./sub/PG_VERSION") differs
        // from the normalized seed key, so this fails unless the op normalizes (the bug this guards).
        let (fp, fl) = put(&mut win, PA, "./sub/PG_VERSION");
        let fd = call(&mut host, h, &mut win, FS_OPEN, &[fp, fl, O_READ, 0]);
        assert!(fd >= 0, "open ./sub/PG_VERSION (normalized) succeeds");
        let n = call(&mut host, h, &mut win, FS_READ, &[fd, RB as i64, 8, 0]);
        assert_eq!(n, 3, "read the seeded 3 bytes");
        assert_eq!(&win[RB..RB + 3], b"17\n", "seeded file contents");
        call(&mut host, h, &mut win, FS_CLOSE, &[fd, 0, 0, 0]);

        // Open the *empty* seeded directory read-only and fsync it — Postgres does this at checkpoint.
        let (dp, dl) = put(&mut win, PA, "sub/mappings");
        let dfd = call(&mut host, h, &mut win, FS_OPEN, &[dp, dl, O_READ, 0]);
        assert!(dfd >= 0, "open a directory read-only (for fsync) succeeds");
        assert_eq!(
            call(&mut host, h, &mut win, FS_SYNC, &[dfd, 0, 0, 0]),
            0,
            "fsync of a directory fd succeeds"
        );
        call(&mut host, h, &mut win, FS_CLOSE, &[dfd, 0, 0, 0]);

        // A seeded directory stats as 0700 (Postgres' `checkDataDir` rejects group/other perms).
        let (sp, sl) = put(&mut win, PA, "sub");
        assert_eq!(
            call(
                &mut host,
                h,
                &mut win,
                FS_STAT,
                &[sp, sl, SB as i64, STATBUF_LEN as i64]
            ),
            0
        );
        let mode = u32::from_le_bytes(win[SB..SB + 4].try_into().unwrap());
        assert_eq!(mode & S_IFMT, S_IFDIR, "seeded 'sub' is a directory");
        assert_eq!(mode & 0o777, 0o700, "in-memory dir reports owner-only 0700");

        // The directory-open path is deliberately narrow: a read-only open of a name that is neither a
        // file nor a *tracked* directory is still `-ENOENT` — it must not resolve arbitrary paths to a
        // directory fd (the over-broad `is_dir` version silently opened file-key prefixes and the root,
        // which broke ordinary `open`s on the general in-memory fs).
        let (np, nl) = put(&mut win, PA, "sub/nope");
        assert_eq!(
            call(&mut host, h, &mut win, FS_OPEN, &[np, nl, O_READ, 0]),
            -ENOENT,
            "read-only open of a non-file, non-tracked-dir path is ENOENT (dir-open stays narrow)"
        );
        // And a create still makes a *file* even where a dir subtree exists alongside (write intent
        // never resolves to a directory fd).
        let (cp, cl) = put(&mut win, PA, "sub/new");
        let cfd = call(
            &mut host,
            h,
            &mut win,
            FS_OPEN,
            &[cp, cl, O_CREATE | O_WRITE, 0],
        );
        assert!(
            cfd >= 0,
            "create under a seeded subtree opens a writable file"
        );
        call(&mut host, h, &mut win, FS_CLOSE, &[cfd, 0, 0, 0]);
    }

    /// `opendir` snapshots entries and `readdir` yields them **sorted**, deterministically, across
    /// several files and a nested directory — and a too-small `readdir` buffer fails closed
    /// (`-EINVAL`) without consuming the entry.
    #[test]
    fn readdir_is_sorted_and_bounded() {
        use super::*;
        use svm_interp::{iface, Host, WindowMem};

        fn call(host: &mut Host, h: i32, win: &mut [u8], op: u32, args: &[i64]) -> i64 {
            let n = win.len();
            let mut wm = WindowMem::new(win, n as u64);
            host.cap_dispatch_slots(iface::HOST_FN, op, h, args, Some(&mut wm))
                .unwrap()[0]
        }
        fn names(host: &mut Host, h: i32, win: &mut [u8]) -> Vec<String> {
            // opendir "." (the root) and drain it.
            win[0] = b'.';
            let dh = call(host, h, win, FS_OPENDIR, &[0, 1, 0, 0]);
            assert!(dh >= 0, "opendir . : {dh}");
            let mut out = Vec::new();
            loop {
                let n = call(host, h, win, FS_READDIR, &[dh, 512, 64, 0]);
                if n == 0 {
                    break;
                }
                assert!(n > 0, "readdir: {n}");
                out.push(String::from_utf8(win[512..512 + n as usize].to_vec()).unwrap());
            }
            call(host, h, win, FS_CLOSEDIR, &[dh, 0, 0, 0]);
            out
        }

        // Build "beta", "alpha", "gamma/x" (gamma implicit) on mem_fs; expect sorted [alpha,beta,gamma].
        let mut host = Host::new();
        let cap = mem_fs();
        let h = (cap.grant)(&mut host, 0);
        let mut win = vec![0u8; 4096];
        for f in ["beta", "alpha", "gamma/x"] {
            win[100..100 + f.len()].copy_from_slice(f.as_bytes());
            let fd = call(
                &mut host,
                h,
                &mut win,
                FS_OPEN,
                &[100, f.len() as i64, O_CREATE | O_WRITE, 0],
            );
            assert!(fd >= 0);
            call(&mut host, h, &mut win, FS_CLOSE, &[fd, 0, 0, 0]);
        }
        assert_eq!(
            names(&mut host, h, &mut win),
            vec!["alpha", "beta", "gamma"]
        );

        // A readdir into a 2-byte buffer can't hold "alpha" (5) → -EINVAL, entry not consumed.
        win[0] = b'.';
        let dh = call(&mut host, h, &mut win, FS_OPENDIR, &[0, 1, 0, 0]);
        assert_eq!(
            call(&mut host, h, &mut win, FS_READDIR, &[dh, 512, 2, 0]),
            -EINVAL,
            "a too-small readdir buffer fails closed"
        );
        // The entry survived: a full-size read still returns it.
        assert_eq!(
            call(&mut host, h, &mut win, FS_READDIR, &[dh, 512, 64, 0]),
            5
        );
    }
}
