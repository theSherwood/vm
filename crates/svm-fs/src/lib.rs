//! **In-memory `fs` capability backend + the shared fs-cap wire protocol.**
//!
//! This crate holds the wasm-safe half of the filesystem capability (`crates/svm-run/src/fs.rs`
//! §7): the op-code/`stat`-layout/path-vetting protocol both backends speak, and the deterministic
//! **in-memory** backend (`mem_fs`) plus its shippable **data-image** format. It depends only on
//! `svm-interp` (`HostFn`/`GuestMem`), so it builds for **wasm** — the browser cdylib mounts a
//! data-image `mem_fs` with no real filesystem. `svm-run` keeps the real-filesystem `host_fs`
//! backend (which pulls in the unix-only JIT/mmap machinery) and wraps these handlers in its
//! `HostCap`; it re-exports this crate's protocol + `mem_fs*` so `svm_run::fs::*` is unchanged.
//!
//! A handler builder returns a `make: impl Fn() -> HostFn` closure — `svm-run` passes it to
//! `HostCap::host_fn`, and the browser cdylib grants the `HostFn` directly on its `svm-interp` Host.

use std::collections::HashMap;
use std::path::{Component, Path};
use std::sync::{Arc, Mutex};
use svm_interp::{GuestMem, HostFn};

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
pub fn stat_bytes(
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
pub fn read_path(mem: Option<&dyn GuestMem>, ptr: i64, len: i64) -> Result<String, i64> {
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
pub fn alloc_fd<T>(open: &mut Vec<Option<T>>) -> usize {
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
pub struct CrashCtl {
    /// Barriers that may still complete before the crash trips; `None` = disarmed (never crash).
    pub countdown: Option<u64>,
    /// Once set, all persistence is frozen.
    pub crashed: bool,
}

impl CrashCtl {
    /// Call at each durability barrier (`msync`/`sync`). Returns `true` if this barrier's write must be
    /// **dropped** — either we have already crashed, or this very barrier trips the crash.
    pub fn barrier(&mut self) -> bool {
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
    /// Current contents as a `(files, dirs)` seed — the exact shape [`encode_image`] serializes and
    /// [`mem_fs_seeded_handler`]/[`mem_fs_seeded_shared`] mount. A file's bytes are its live committed
    /// buffer (an open `fd` shares the same `Arc`, so bytes already `write`n are included); purely
    /// transient state that is not part of a filesystem *image* — open-fd cursors, `opendir` handles,
    /// live mmaps — is dropped. Entries are sorted for a byte-deterministic image (so re-snapshotting an
    /// unchanged store yields identical bytes).
    fn snapshot(&self) -> FsSeed {
        let mut files: Vec<(String, Vec<u8>)> = self
            .files
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.lock().unwrap_or_else(|e| e.into_inner()).clone(),
                )
            })
            .collect();
        files.sort_by(|a, b| a.0.cmp(&b.0));
        let dirs: Vec<String> = self.dirs.iter().cloned().collect(); // BTreeSet ⇒ already sorted
        (files, dirs)
    }

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
pub fn arm_crash(ctl: Option<&mut CrashCtl>, n: i64) -> i64 {
    let Some(c) = ctl else {
        return -EINVAL;
    };
    c.countdown = if n < 0 { None } else { Some(n as u64) };
    c.crashed = false;
    0
}

/// A `make` builder for the in-memory backend (`svm-run` wraps it in a `HostCap`; the browser grants
/// the `HostFn` directly). `crashy` enables the **test-only** crash-injection op ([`FS_CRASH_ARM`]).
pub fn mem_fs_handler(crashy: bool) -> impl Fn() -> HostFn + Send + Sync + 'static {
    move || {
        let mut st = MemFsState {
            crash: crashy.then(CrashCtl::default),
            ..MemFsState::default()
        };
        Box::new(
            move |op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>| {
                Ok(vec![st.handle(op, args, mem)])
            },
        ) as HostFn
    }
}

/// A `make` builder for a **pre-seeded** in-memory backend (a mounted data image): `files` maps a
/// path to its bytes, `dirs` names directories that must exist even when empty. Each grant clones the
/// seed fresh, so a run's writes never leak back into it (deterministic re-runs).
pub fn mem_fs_seeded_handler(
    files: Vec<(String, Vec<u8>)>,
    dirs: Vec<String>,
) -> impl Fn() -> HostFn + Send + Sync + 'static {
    let files = Arc::new(files);
    let dirs = Arc::new(dirs);
    move || {
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
    }
}

/// A live handle onto a store mounted by [`mem_fs_seeded_shared`], letting the caller serialize the
/// **current** filesystem back out — e.g. to persist a browser Postgres session across page reloads
/// (snapshot the data dir, stash the image, reboot from it next visit). Cloneable; every clone observes
/// the same live store. The guest that owns the mount runs single-threaded, so a snapshot taken while it
/// is suspended (parked at a stdin read, between queries) is a quiescent, crash-consistent point-in-time
/// image — exactly the state Postgres' startup recovery expects to replay.
#[derive(Clone)]
pub struct MemFsHandle(Arc<Mutex<MemFsState>>);

impl MemFsHandle {
    /// The current filesystem as a `(files, dirs)` seed (see [`MemFsState::snapshot`]).
    pub fn seed(&self) -> FsSeed {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).snapshot()
    }

    /// The current filesystem serialized straight to a shippable [`encode_image`] blob — the bytes a
    /// caller persists and later re-mounts via [`decode_image`] + [`mem_fs_seeded_shared`].
    pub fn image(&self) -> Vec<u8> {
        let (files, dirs) = self.seed();
        encode_image(&files, &dirs)
    }
}

/// Like [`mem_fs_seeded_handler`] but returns the `HostFn` **and** a [`MemFsHandle`] onto its state, so
/// the mount can be snapshotted back out later (the persistent-session persistence path). Unlike the
/// `make: impl Fn() -> HostFn` builders — which re-seed a fresh store on every grant — this grants
/// **one** live store shared between the handler and the handle through an `Arc<Mutex<..>>`, locked per
/// op (uncontended in the single-threaded browser). The deterministic, snapshot-free one-shot path keeps
/// [`mem_fs_seeded_handler`].
pub fn mem_fs_seeded_shared(
    files: Vec<(String, Vec<u8>)>,
    dirs: Vec<String>,
) -> (HostFn, MemFsHandle) {
    let mut st = MemFsState::default();
    for (p, data) in &files {
        st.files.insert(norm(p), Arc::new(Mutex::new(data.clone())));
    }
    for d in &dirs {
        st.dirs.insert(norm(d));
    }
    let shared = Arc::new(Mutex::new(st));
    let handle = MemFsHandle(shared.clone());
    let hostfn: HostFn = Box::new(
        move |op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>| {
            Ok(vec![shared
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .handle(op, args, mem)])
        },
    );
    (hostfn, handle)
}

/// A filesystem seed: `(files as (relative-path, bytes), directory relative-paths)`. The material both
/// [`mem_fs_seeded`] mounts and [`encode_image`] serializes.
pub type FsSeed = (Vec<(String, Vec<u8>)>, Vec<String>);

/// Walk a host directory into a [`FsSeed`] — every regular file's `(relative-path, bytes)` plus every
/// directory's relative path (so empty ones survive). Symlinks are followed. The raw material for both
/// [`mem_fs_from_host_dir`] and [`encode_image`] (build a shippable data image once).
pub fn read_host_dir(root: &Path) -> std::io::Result<FsSeed> {
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
    Ok((files, dirs))
}

const IMAGE_MAGIC: &[u8; 8] = b"SVMFSIM1";

/// Serialize a `(files, dirs)` seed into a **self-contained data image** — a flat, portable byte blob a
/// demo ships and mounts with [`mem_fs_from_archive`] (no host filesystem needed, e.g. in the browser).
/// Format (all little-endian): magic `SVMFSIM1`; `u32` dir count, then each dir `u32 len + path`; `u32`
/// file count, then each file `u32 path-len + path + u64 data-len + data`. Paths are stored verbatim
/// (normalization happens at mount time, as for any seed).
pub fn encode_image(files: &[(String, Vec<u8>)], dirs: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(IMAGE_MAGIC);
    out.extend_from_slice(&(dirs.len() as u32).to_le_bytes());
    for d in dirs {
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d.as_bytes());
    }
    out.extend_from_slice(&(files.len() as u32).to_le_bytes());
    for (p, data) in files {
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        out.extend_from_slice(p.as_bytes());
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
        out.extend_from_slice(data);
    }
    out
}

/// Parse a [`encode_image`] blob back into a [`FsSeed`]. `Err` on a bad magic or a truncated/oversized
/// field (fail-closed — a corrupt image never yields a partial mount).
pub fn decode_image(bytes: &[u8]) -> Result<FsSeed, String> {
    let mut p = 0usize;
    let take = |p: &mut usize, n: usize| -> Result<&[u8], String> {
        let end = p.checked_add(n).ok_or("image: length overflow")?;
        let s = bytes.get(*p..end).ok_or("image: truncated")?;
        *p = end;
        Ok(s)
    };
    let u32at = |p: &mut usize| -> Result<usize, String> {
        Ok(u32::from_le_bytes(take(p, 4)?.try_into().unwrap()) as usize)
    };
    let u64at = |p: &mut usize| -> Result<usize, String> {
        let v = u64::from_le_bytes(take(p, 8)?.try_into().unwrap());
        usize::try_from(v).map_err(|_| "image: entry too large".to_string())
    };
    if take(&mut p, 8)? != IMAGE_MAGIC {
        return Err("image: bad magic".into());
    }
    let n_dirs = u32at(&mut p)?;
    let mut dirs = Vec::with_capacity(n_dirs);
    for _ in 0..n_dirs {
        let len = u32at(&mut p)?;
        let s = std::str::from_utf8(take(&mut p, len)?).map_err(|_| "image: non-UTF-8 dir")?;
        dirs.push(s.to_string());
    }
    let n_files = u32at(&mut p)?;
    let mut files = Vec::with_capacity(n_files);
    for _ in 0..n_files {
        let len = u32at(&mut p)?;
        let s = std::str::from_utf8(take(&mut p, len)?).map_err(|_| "image: non-UTF-8 path")?;
        let dlen = u64at(&mut p)?;
        files.push((s.to_string(), take(&mut p, dlen)?.to_vec()));
    }
    Ok((files, dirs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat `GuestMem` over a `Vec<u8>` — enough to drive the fs ops (they only read paths / read
    /// write-payloads / write read-results within `[0, len)`).
    struct VecMem(Vec<u8>);
    impl GuestMem for VecMem {
        fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
            let end = (ptr as usize).checked_add(len as usize)?;
            self.0.get(ptr as usize..end).map(<[u8]>::to_vec)
        }
        fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
            let end = (ptr as usize).checked_add(data.len())?;
            self.0.get_mut(ptr as usize..end)?.copy_from_slice(data);
            Some(())
        }
    }

    /// `mem_fs_seeded_shared` snapshots the **live** store — writes and removes made through the granted
    /// `HostFn` after the mount show up in `MemFsHandle::image`, and the image round-trips through
    /// `decode_image` back to a mountable seed. This is the persistence hinge: a Postgres session's data
    /// dir, mutated by DDL/DML, must serialize back out exactly.
    #[test]
    fn shared_handle_snapshots_live_writes() {
        // Seed: one existing file + one empty dir (the shapes an initdb tree has).
        let seed_files = vec![("base/1".to_string(), b"seed".to_vec())];
        let seed_dirs = vec!["pg_wal".to_string()];
        let (mut fs, handle) = mem_fs_seeded_shared(seed_files, seed_dirs);

        // The seed is visible immediately, before any op.
        let (f0, d0) = handle.seed();
        assert_eq!(f0, vec![("base/1".to_string(), b"seed".to_vec())]);
        assert_eq!(d0, vec!["pg_wal".to_string()]);

        // Lay out guest memory: path "base/2" at 0, payload "hello" at 16.
        let mut mem = VecMem(vec![0u8; 32]);
        mem.0[..6].copy_from_slice(b"base/2");
        mem.0[16..21].copy_from_slice(b"hello");
        let call = |fs: &mut HostFn, op: u32, args: &[i64], mem: &mut VecMem| -> i64 {
            fs(op, args, Some(mem)).expect("host fn")[0]
        };

        // Create + write a new file through the granted handler (O_CREATE|O_WRITE).
        let fd = call(&mut fs, FS_OPEN, &[0, 6, O_CREATE | O_WRITE], &mut mem);
        assert!(fd >= 3, "fd = {fd}");
        assert_eq!(call(&mut fs, FS_WRITE, &[fd, 16, 5], &mut mem), 5);
        assert_eq!(call(&mut fs, FS_CLOSE, &[fd], &mut mem), 0);

        // Remove the seeded file (path "base/1" reuses the same 6-byte slot layout).
        mem.0[..6].copy_from_slice(b"base/1");
        assert_eq!(call(&mut fs, FS_REMOVE, &[0, 6], &mut mem), 0);

        // Snapshot → the write is captured and the removal propagated.
        let image = handle.image();
        let (files, dirs) = decode_image(&image).expect("round-trips");
        assert_eq!(files, vec![("base/2".to_string(), b"hello".to_vec())]);
        assert_eq!(dirs, vec!["pg_wal".to_string()]);

        // Re-mounting the image reproduces the same live state (the persistence loop closes).
        let (_fs2, handle2) = mem_fs_seeded_shared(files, dirs);
        assert_eq!(handle2.seed(), handle.seed());

        // Snapshotting an unchanged store is byte-identical (deterministic, sorted output).
        assert_eq!(handle.image(), handle.image());
    }
}
