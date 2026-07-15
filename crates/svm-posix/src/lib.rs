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
//! Scope: `write` / `read` / `malloc` / `free` / `exit`, plus `open` / `close` / `lseek` over an
//! in-memory filesystem (a `path → bytes` memfs) with a host-side fd table. `malloc` is a first-fit
//! free list over a configured window-heap region. Still to come (POSIX.md §6): `stat`/`readdir`,
//! signals, and `fork`/`exec`. Pure computation (`strlen`, `snprintf`, `math`, …) is **guest code**,
//! not a cap — it needs no authority (POSIX.md §1).
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

/// Negative errnos this personality returns (Linux values, so a guest's `<errno.h>` agrees).
const ENOENT: i64 = -2; // no such file (open without O_CREAT)
const EBADF: i64 = -9; // an op on an fd this personality does not serve
const EINVAL: i64 = -22; // bad argument (whence, non-UTF-8 path, negative seek)

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
        _ => return None,
    };
    Some(ResolvedCap {
        type_id: iface::HOST_FN,
        op,
    })
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
        // Bump a fresh block from the high-water mark (already `ALIGN`-aligned by construction).
        let ptr = (self.heap_next + (ALIGN - 1)) & !(ALIGN - 1);
        match ptr.checked_add(want) {
            Some(end) if end <= self.heap_end => {
                self.heap_next = end;
                self.allocated.insert(ptr, want);
                ptr as i64
            }
            _ => 0, // out of heap → NULL
        }
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
