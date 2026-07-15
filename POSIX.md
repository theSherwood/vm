# POSIX personality — libc as host capabilities

> Status: **design + first spike.** `svm-posix` provides the first ops (`write` / `read` /
> `malloc` / `free` / `exit`) as a `HostFn` capability, differential-tested on both backends.
> This doc is the plan the rest of the surface fills in. Update the **ABI table** and the
> **Status** as ops land.

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
| — | `stat/fstat/readdir/getcwd/chdir/unlink` | / `-errno` | memfs + host fd table | todo |
| — | `getenv/setenv/environ` | `-> ptr \| 0` | host env map | todo |
| — | `signal/sigaction/kill` | doorbell (§9 L0) | host signal state, checked at command boundaries | todo |
| — | `pipe/dup/dup2/fcntl` | `-> fd \| -errno` | `Pipe` cap + host fd table | todo |
| — | `time/clock_gettime` | `-> t` | `Clock` cap | todo |
| — | `fork/execve/waitpid` | Stage 3 | `Instantiator` / clone (§7) | parked |
| — | `strlen/memcpy/snprintf/qsort/ctype/math` | pure | **guest code** (no cap) | n/a |

## 6. Roadmap

1. **Spike (done):** `svm-posix` crate — `write`/`read`/`malloc`/`free`/`exit` as a `HostFn`,
   differential interp↔JIT (`svm-posix` tests). Proves the arena-in-window + host-bookkeeping
   model and the cross-backend parity.
2. **Named-import binding:** wire `svm_posix::resolve` into `resolve_capability_imports` and
   run a C `main` (via chibicc) that calls `write`/`malloc` through named imports — the real
   linking path, not hand-written cap.calls.
3. **fs + fd table:** `open`/`read`/`write`/`close`/`stat`/`readdir` over the existing fs ops,
   with a host-side fd table; a real free-list allocator.
4. **A first shell:** BusyBox `ash` (fork-less) at Stage 0/2 — `sh -c`, builtins, `ls | grep`
   via the `Pipe` cap. This is the playground target.
5. **Signals (L0), env, time**, then Stage 3 (`fork`/`exec`) on top of `Instantiator` / clone.

Testing follows the repo standard: every op is an interp ↔ bytecode ↔ JIT differential
(errno paths included), because a `HostFn` dispatches through the same `cap_dispatch_slots`
the JIT's `cap.call` thunk calls — parity for free.
