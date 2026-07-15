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
//! Scope: the first spike — `write` / `read` / `malloc` / `free` / `exit`. `malloc` is a bump
//! allocator over a configured heap region (`free` is a no-op); a real free list, the fs + fd table,
//! signals, and `fork`/`exec` are the roadmap (POSIX.md §6). Pure computation (`strlen`, `snprintf`,
//! `math`, …) is **guest code**, not a cap — it needs no authority (POSIX.md §1).
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

/// `-EBADF` — a `write`/`read` on an fd this personality does not serve.
const EBADF: i64 = -9;

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
            _ => Err(Trap::CapFault),
        }
    })
}

impl Inner {
    /// `write(fd, buf, len) -> n | -errno`: copy `len` bytes from the window at `buf` to the captured
    /// `fd` sink (`1` stdout / `2` stderr), returning the count. Other fds are `-EBADF` (the fs fd
    /// table is a follow-up).
    fn write(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let fd = *args.first().ok_or(Trap::Malformed)?;
        let buf = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let len = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as u64;
        let sink = match fd {
            1 => &mut self.stdout,
            2 => &mut self.stderr,
            _ => return Ok(vec![EBADF]),
        };
        if len == 0 {
            return Ok(vec![0]);
        }
        let data = mem.read_bytes(buf, len).ok_or(Trap::Malformed)?;
        sink.extend_from_slice(&data);
        Ok(vec![len as i64])
    }

    /// `read(fd, buf, len) -> n | -errno`: drain up to `len` bytes of preloaded stdin (`fd` `0`) into
    /// the window at `buf`, returning the count (`0` at EOF). Other fds are `-EBADF`.
    fn read(&mut self, args: &[i64], mem: Option<&mut dyn GuestMem>) -> Result<Vec<i64>, Trap> {
        let mem = mem.ok_or(Trap::Malformed)?;
        let fd = *args.first().ok_or(Trap::Malformed)?;
        let buf = *args.get(1).ok_or(Trap::Malformed)? as u64;
        let len = (*args.get(2).ok_or(Trap::Malformed)?).max(0) as usize;
        if fd != 0 {
            return Ok(vec![EBADF]);
        }
        let avail = &self.stdin[self.stdin_pos.min(self.stdin.len())..];
        let n = len.min(avail.len());
        let chunk = avail[..n].to_vec();
        mem.write_bytes(buf, &chunk).ok_or(Trap::Malformed)?;
        self.stdin_pos += n;
        Ok(vec![n as i64])
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

    #[test]
    fn resolve_binds_libc_names() {
        // The §7 name → (HOST_FN, op) map a linker uses to bind a shell's libc imports.
        assert_eq!(resolve("malloc").map(|c| c.op), Some(OP_MALLOC));
        assert_eq!(resolve("posix.write").map(|c| c.op), Some(OP_WRITE));
        assert_eq!(resolve("_exit").map(|c| c.op), Some(OP_EXIT));
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
