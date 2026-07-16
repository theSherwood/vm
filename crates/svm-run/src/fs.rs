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
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use svm_interp::{GuestMem, HostFn, HostFnRegion, RegionMinter};

// The shared fs-cap wire protocol, the in-memory backend, and the shippable data-image format live in
// the wasm-safe `svm-fs` crate — so the browser cdylib can mount a `mem_fs` without the unix-only JIT
// that `svm-run` pulls in. Re-export them so `svm_run::fs::*` is unchanged; the real-filesystem
// `host_fs` backend and the `HostCap` wrappers (svm-run's capability type) stay here.
pub use svm_fs::*;

/// A deterministic **in-memory** filesystem capability (fresh, empty state per host). The hermetic
/// default for tests and differential runs.
pub fn mem_fs() -> HostCap {
    HostCap::host_fn(0, svm_fs::mem_fs_handler(false))
}

/// Like [`mem_fs`] but with the **test-only crash-injection** controller enabled (the [`FS_CRASH_ARM`]
/// op becomes live). Used to prove crash-consistency of a mapped store — see `demo_lmdb_crash_recovery`.
/// Never grant this to a real guest: a tripped crash freezes the store (a self-inflicted DoS on the
/// holder's own fs, no host effect, but pointless outside a test).
pub fn mem_fs_crashy() -> HostCap {
    HostCap::host_fn(0, svm_fs::mem_fs_handler(true))
}

/// A **pre-seeded** in-memory filesystem: `files` maps a normalized relative path to its contents, and
/// `dirs` names directories that must exist even with no files under them. Each host grant gets a fresh
/// clone of the seed, so a run's writes never leak back into it — re-runs are deterministic. This is how
/// a demo with no real filesystem (e.g. the browser) mounts a data-dir image on the `fs` cap.
pub fn mem_fs_seeded(files: Vec<(String, Vec<u8>)>, dirs: Vec<String>) -> HostCap {
    HostCap::host_fn(0, svm_fs::mem_fs_seeded_handler(files, dirs))
}

/// Walk a host directory into a [`mem_fs_seeded`] capability — every regular file's bytes keyed by its
/// path relative to `root`, plus every directory (so empty ones survive). For building a demo data
/// image from an on-disk cluster; the resulting cap has no further tie to the host filesystem.
pub fn mem_fs_from_host_dir(root: &Path) -> std::io::Result<HostCap> {
    let (files, dirs) = svm_fs::read_host_dir(root)?;
    Ok(mem_fs_seeded(files, dirs))
}

/// Mount a [`encode_image`] data-image blob as an in-memory `fs` cap — the browser/demo path (a shipped
/// image, no host filesystem). Decode + [`mem_fs_seeded`].
pub fn mem_fs_from_archive(bytes: &[u8]) -> Result<HostCap, String> {
    let (files, dirs) = svm_fs::decode_image(bytes)?;
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

    /// A data image round-trips through [`encode_image`]/[`decode_image`] byte-exact, and mounts via
    /// [`mem_fs_from_archive`] into a working fs — the shippable-artifact path (no host filesystem).
    #[test]
    fn data_image_roundtrip_and_mount() {
        use super::*;
        use svm_interp::{iface, Host, WindowMem};

        let files = vec![
            ("PG_VERSION".to_string(), b"17\n".to_vec()),
            ("base/1/1259".to_string(), vec![0u8; 300]),
            ("global/pg_control".to_string(), b"\x01\x02\x03".to_vec()),
        ];
        let dirs = vec!["base/1".to_string(), "pg_logical/mappings".to_string()];

        let img = encode_image(&files, &dirs);
        let (df, dd) = decode_image(&img).expect("decode");
        assert_eq!(df, files, "files round-trip byte-exact");
        assert_eq!(dd, dirs, "dirs round-trip");
        assert!(decode_image(b"nope").is_err(), "bad magic fails closed");
        assert!(
            decode_image(&img[..img.len() - 5]).is_err(),
            "truncation fails closed"
        );

        // Mount the image and read a file back through the cap.
        let cap = mem_fs_from_archive(&img).expect("mount archive");
        let mut host = Host::new();
        let h = (cap.grant)(&mut host, 0);
        let mut win = vec![0u8; 4096];
        win[..10].copy_from_slice(b"PG_VERSION");
        let mut wm = WindowMem::new(&mut win, 4096);
        let fd = host
            .cap_dispatch_slots(
                iface::HOST_FN,
                FS_OPEN,
                h,
                &[0, 10, O_READ, 0],
                Some(&mut wm),
            )
            .unwrap()[0];
        assert!(fd >= 0, "open a file from the mounted image");
        let n = host
            .cap_dispatch_slots(iface::HOST_FN, FS_READ, h, &[fd, 512, 8, 0], Some(&mut wm))
            .unwrap()[0];
        assert_eq!(n, 3);
        assert_eq!(&win[512..515], b"17\n", "file contents from the image");
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
