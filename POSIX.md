# POSIX personality — libc as host capabilities

> Status: **core surface landed.** `svm-posix` provides ops 0–20 (stdio, `malloc`/`free`,
> `exit`, the memfs + fd table, cwd, env, argv, and the Stage-1 `exec` surface) as a `HostFn`
> capability, differential-tested on both backends. A compiled-C shell runs on it
> (`crates/svm/tests/c_shell.rs`) and dispatches external commands to spawned confined
> children (`stage1_posix_spawn.rs`; the spawn work is tracked in `STAGE1.md`). Update the
> **ABI table** and the **Status** as ops land.

## 1. The thesis: libc is not one thing

Running a shell (BusyBox `ash` → Bash) needs a libc. We do **not** bake a libc into the VM
or the guest as a fixed, trusted blob. The insight is that "libc" splits along one line —
**authority** — and only one side is naturally a host capability:

- **The authority / OS surface** — `open`/`read`/`write`/`lseek`/`stat`/`readdir`, `mmap`,
  `fork`/`exec`/`waitpid`, `signal`, `time`, `exit`, `pipe`. These carry authority, so they
  are **host-provided capabilities**, reached only through granted, masked, type-checked
  handles (§7). Most already exist as first-class caps (`Stream`, `Memory`, the fs ops,
  `Instantiator`, `Pipe`, `Clock`, `Exit`); the rest are `HostFn` ops. **This is the part
  that must never be baked in — and it isn't.**
- **The pure-computation bulk** — `strlen`/`memcpy`/`snprintf`-formatting, `qsort`, `ctype`,
  `math`, and malloc's *free-list logic*. These carry **no authority**. Whether they are
  host caps or ordinary guest code is a **performance / binary-size choice, not a security
  one**: guest code is re-verified and untrusted like any guest — it is *not* in the TCB.
  A host cap for `strlen` is a boundary crossing for a loop, so the default is to keep pure
  compute as **guest code** (compiled from the C source), and reach for a host cap only when
  binary size or startup demands it.

"Baked in" means *in the TCB*. Neither guest code nor a `HostFn` is in the TCB. The whole
personality — every byte of libc semantics — lives outside the escape-TCB match, exactly the
boundary DESIGN.md §7 draws.

## 2. The mechanism already exists: §7 named imports → `HostFn`

`svm-wasi` is the working template (a 2-op WASI shim); `svm-posix` generalizes it.

1. The shell (C) compiles to SVM IR with its libc calls left as **unresolved named imports**
   — `CallImport { "env.malloc" }`, `"posix.open"`, … (chibicc / `svm-llvm` emit these for
   any unresolved symbol).
2. At **load**, `svm_ir::resolve_imports` (driven by `svm_run::resolve_capability_imports`)
   binds each name to a `(type_id, op)` on a capability handle and lowers `CallImport` →
   `cap.call`. `svm_posix::resolve(name)` supplies the name → `(HOST_FN, op)` map.
3. A `HostFn` handler implements each op **host-side**, reading/writing the guest window
   through the masked `GuestMem`. All names share **one** `HOST_FN` handle; the op number
   distinguishes the call (svm-wasm/chibicc thread a single capability handle).

Nothing here touches the verifier or the confinement lowering. A `HostFn` is untrusted host
code reached only through a masked handle — a translation or personality bug is a clean
capability error, never an escape.

## 3. Where the state lives: bytes in the window, bookkeeping host-side

The elegant split for the stateful ops (`malloc`, `stdio`, the fd table):

- **Bytes live in the guest window.** A `malloc`'d buffer, a `FILE`'s scratch, an `iovec`
  target — all are ordinary window memory the guest reads and writes at **native speed** (no
  crossing per byte). `malloc(n)` returns a **window offset**.
- **Bookkeeping lives host-side**, in the `HostFn`'s state: the allocator's free list, the
  fd table (fd → `Stream`/`Pipe`/fs handle), `FILE` buffering, `errno`. This is the small,
  swappable part; it never enters the guest's address space, so the guest cannot corrupt it.

The allocator manages a **heap region of the guest window** the embedder configures at grant
(`[heap_base, heap_end)`): a first-fit **free list** — `malloc` reuses a freed block (splitting
off any remainder) before bumping the high-water mark, and `free` returns a block for reuse
(coalescing adjacent frees is a follow-up). The **fs** is an in-memory `path → bytes` map (a
memfs) with a host-side fd table (`open`/`close`/`read`/`write`/`lseek`); it keeps the
personality self-contained for the playground, and a native embedder routing to a real `fs`
cap is a follow-up.

## 4. One ABI, two bindings — how this unifies with "personality = guest library"

PROCESS.md frames personalities as guest libraries; this doc frames libc as host caps. The
§7 named-import ABI makes these **the same interface with different bindings**: a name can
resolve to a **host** cap (`HostFn` — fast, the playground path) *or* to a cap a **parent
serves** (an endpoint — the self-similar / interposition path, PROCESS.md Stage 2.5). The
shell's IR is identical; only the resolver's target differs. So the durable decision is to
**pin the ABI** — the function list and each function's shape — and bind it host-side now,
guest-serve the same ABI later.

**The handle binds by name too — no powerbox slot (PROCESS.md S15).** The personality is a
per-domain **singleton**, so its handle is supplied by the resolver, not threaded by the
guest: each libc import's handle operand is a `ConstI32` **placeholder** patched at resolve
(`svm_ir::Resolved::CapBound`, via `svm_posix::resolve_bound(handle)` — grant first, then
resolve; DESIGN.md §7's "late binding is the general form of the powerbox"). Consequences:
the guest's libc has **real C signatures** (`open(path, flags)`, `getenv(name)`, `malloc(n)`
— the NUL→`(ptr,len)` adaptation is a thin guest wrapper); the module's **import section is
the discoverable capability manifest** (explicit names + signatures, fail-closed — never a
silent slot numbering agreed out-of-band); and nothing about the personality touches the
fixed 8-slot `_start`, which S15 retires. Capabilities with **many** live objects (streams,
regions, pipe ends) keep the handle a call-site operand — resolver-bound handles are the
singleton case, not a replacement for first-class handles.

## 5. The ABI (POSIX subset for a fork-less shell)

Op numbers on the shared `HOST_FN` handle. `ptr`/`buf` are **window offsets**; `-errno` on
failure (`< 0`), a `>= 0` count / handle on success (except `malloc`, which returns `0` for
"no memory", the C `NULL`). Pure-compute functions are **guest code** (no cap) and are listed
only to mark the boundary.

| # | Function | Shape | Backed by | Status |
|---|----------|-------|-----------|--------|
| 0 | `write(fd, buf, len)` | `-> n \| -errno` | host sinks / fd table → `Stream`/`Pipe` | **done (spike)** — fd 1/2 → captured stdout/stderr |
| 1 | `read(fd, buf, len)` | `-> n \| -errno` | host sinks / fd table → `Stream`/`Pipe` | **done (spike)** — fd 0 → preloaded stdin |
| 2 | `malloc(size)` | `-> ptr \| 0` | window-heap arena (host bookkeeping) | **done** — first-fit free list |
| 3 | `free(ptr)` | `-> 0` | window-heap arena | **done** — reclaims for reuse (no coalescing yet) |
| 4 | `exit(code)` | `noreturn` | `Trap::Exit` (→ `Exit` cap) | **done** |
| 5 | `open(path, len, flags)` | `-> fd \| -errno` | memfs + host fd table | **done** — `O_CREAT`/`O_TRUNC`/`O_APPEND`, `-ENOENT` |
| 6 | `close(fd)` | `-> 0 \| -errno` | host fd table | **done** |
| 7 | `lseek(fd, off, whence)` | `-> pos \| -errno` | host fd table | **done** — `SEEK_SET`/`CUR`/`END` |
| 8 | `unlink(path, len)` | `-> 0 \| -errno` | memfs | **done** — `-ENOENT` if absent (aka `remove`) |
| 9 | `getcwd(buf, size)` | `-> buf \| -errno` | host cwd | **done** — NUL-terminated, `-ERANGE`/`-EINVAL` |
| 10 | `chdir(path, len)` | `-> 0 \| -errno` | host cwd | **done** — flat memfs, no existence check yet |
| 11 | `getenv(name, len)` | `-> ptr \| 0` | host env map | **done** — stable NUL-terminated `char*` in arena |
| 12 | `setenv(name, nlen, val, vlen, overwrite)` | `-> 0 \| -errno` | host env map | **done** — invalidates `getenv` cache |
| 13 | `stat(path, len, statbuf)` | `-> 0 \| -errno` | memfs | **done** — minimal `{ st_mode, st_size }`; `S_IFREG`/`S_IFDIR`, `-ENOENT` (aka `lstat`) |
| 14 | `opendir(path, len)` | `-> dir \| -errno` | memfs | **done** — snapshots immediate children; `-ENOTDIR`/`-ENOENT` |
| 15 | `readdir(dir, buf, cap)` | `-> namelen \| 0 \| -errno` | dir stream | **done** — NUL-terminated name; `0` at end, `-ERANGE`/`-EBADF` |
| 16 | `closedir(dir)` | `-> 0 \| -errno` | dir stream | **done** — `-EBADF` on a stale handle |
| 17 | `argc()` | `-> n` | host arg vector | **done** — personality ext. (the `sh -c` path; `argv[0]` = program name) |
| 18 | `argv(i, buf, cap)` | `-> len \| -errno` | host arg vector | **done** — NUL-terminated arg `i`; `-EINVAL`/`-ERANGE` |
| 19 | `exec_lookup(name, len)` | `-> module \| -1` | host PATH registry (`register_command`) | **done** — Stage 1 exec (STAGE1.md §5); the spawn itself is the shell's `Instantiator` op 13 + `join` |
| 20 | `exec_stdout()` | `-> stream` | host stdout `Stream` | **done** — the handle the shell re-grants to a child under the name `"stdout"` |
| — | `fstat/environ` | / `-errno` | memfs + host fd table | todo |
| — | `signal/sigaction/kill` | doorbell (§9 L0) | host signal state, checked at command boundaries | todo |
| — | `pipe/dup/dup2/fcntl` | `-> fd \| -errno` | `Pipe` cap + host fd table | todo |
| — | `time/clock_gettime` | `-> t` | `Clock` cap | todo |
| — | `fork/execve/waitpid` | Stage 3 | `Instantiator` / clone (§7) | partial — spawn+wait landed as `Instantiator` op 13 + `join` (Stage 1, STAGE1.md; ops 19/20 above); `fork` itself parked |
| — | `strlen/memcpy/snprintf/qsort/ctype/math` | pure | **guest code** (no cap) | n/a |

## 6. Roadmap

1. **Spike (done):** `svm-posix` crate — `write`/`read`/`malloc`/`free`/`exit` as a `HostFn`,
   differential interp↔JIT (`svm-posix` tests). Proves the arena-in-window + host-bookkeeping
   model and the cross-backend parity.
2. **Named-import binding (done):** a real C `main` (via chibicc) links its libc calls to the
   personality through named imports — the real linking path, not hand-written cap.calls
   (`crates/svm/tests/c_posix.rs`). Bound in the §7 **general form**: `resolve_bound` supplies
   the handle at resolve (`Resolved::CapBound`), so the guest libc has real C signatures and
   no powerbox slot (§4 above; the fixed-`_start` retirement is PROCESS.md S15).
3. **fs + fd table (done):** `open`/`read`/`write`/`close`/`stat`/`readdir` over the memfs,
   with a host-side fd table (ops 5–16 above); a real free-list allocator.
4. **A first shell (done — a compiled-C shell, not BusyBox):** fork-less at Stage 0 — `sh -c`,
   builtins, redirection, pipelines (staged through memfs temp files; `ls | grep` via the
   `Pipe` cap is still open) — `crates/svm/tests/c_shell.rs`. This is the playground target.
5. **Signals (L0), time** (env landed — ops 11/12), then Stage 3 (`fork`/`exec`) on top of
   `Instantiator` / clone. The `exec` half landed first as Stage 1 spawn — ops 19/20 plus the
   shell's own `Instantiator` op 13 + `join` — tracked in `STAGE1.md`; `fork` remains.

Testing follows the repo standard: every op is an interp ↔ bytecode ↔ JIT differential
(errno paths included), because a `HostFn` dispatches through the same `cap_dispatch_slots`
the JIT's `cap.call` thunk calls — parity for free.
