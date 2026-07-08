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

pub const O_READ: i64 = 1;
pub const O_WRITE: i64 = 2;
pub const O_APPEND: i64 = 4;
pub const O_TRUNC: i64 = 8;
pub const O_CREATE: i64 = 16;

pub const ENOENT: i64 = 2;
pub const EBADF: i64 = 9;
pub const EACCES: i64 = 13;
pub const EFAULT: i64 = 14;
pub const EINVAL: i64 = 22;

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

/// One open file: a shared byte buffer (kept alive independently of the name table, so a `remove`
/// of an open file behaves POSIX-like — the data survives until the last close) + cursor + mode.
struct MemOpen {
    data: Arc<Mutex<Vec<u8>>>,
    pos: usize,
    readable: bool,
    writable: bool,
    append: bool,
}

#[derive(Default)]
struct MemFsState {
    files: HashMap<String, Arc<Mutex<Vec<u8>>>>,
    open: Vec<Option<MemOpen>>,
}

impl MemFsState {
    fn handle(&mut self, op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>) -> i64 {
        let mut mem = mem;
        let a = |i: usize| args.get(i).copied().unwrap_or(0);
        match op {
            FS_OPEN => {
                let path = match read_path(mem.as_deref(), a(0), a(1)) {
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
                let fd = self.open.iter().position(|s| s.is_none()).unwrap_or({
                    self.open.push(None);
                    self.open.len() - 1
                });
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
                let path = match read_path(mem.as_deref(), a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                if self.files.remove(&path).is_none() {
                    return -ENOENT;
                }
                0
            }
            FS_RENAME => {
                let from = match read_path(mem.as_deref(), a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let to = match read_path(mem.as_deref(), a(2), a(3)) {
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
                // Memory is always "durable" here — validate the fd, nothing to flush.
                let Some(Some(_)) = self.open.get(a(0) as usize) else {
                    return -EBADF;
                };
                0
            }
            _ => -EINVAL,
        }
    }
}

/// A deterministic **in-memory** filesystem capability (fresh, empty state per host). The hermetic
/// default for tests and differential runs.
pub fn mem_fs() -> HostCap {
    HostCap::host_fn(0, || {
        let mut st = MemFsState::default();
        Box::new(
            move |op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>| {
                Ok(vec![st.handle(op, args, mem)])
            },
        ) as HostFn
    })
}

struct HostOpen {
    file: std::fs::File,
    readable: bool,
    writable: bool,
}

struct HostFsState {
    root: PathBuf,
    open: Vec<Option<HostOpen>>,
}

impl HostFsState {
    fn handle(&mut self, op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>) -> i64 {
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
                let fd = self.open.iter().position(|s| s.is_none()).unwrap_or({
                    self.open.push(None);
                    self.open.len() - 1
                });
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
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return -EBADF;
                };
                match o.file.sync_all() {
                    Ok(()) => 0,
                    Err(e) => -io_errno(&e),
                }
            }
            _ => -EINVAL,
        }
    }
}

fn io_errno(e: &std::io::Error) -> i64 {
    e.raw_os_error().map(|c| c as i64).unwrap_or(EINVAL)
}

/// The **real** filesystem, attenuated to `root` (relative paths only; `..`/absolute refused). The
/// guest sees exactly the subtree the embedder granted — nothing else is nameable.
pub fn host_fs(root: PathBuf) -> HostCap {
    HostCap::host_fn(0, move || {
        let mut st = HostFsState {
            root: root.clone(),
            open: Vec::new(),
        };
        Box::new(
            move |op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>| {
                Ok(vec![st.handle(op, args, mem)])
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
}
