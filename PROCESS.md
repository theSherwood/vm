# Processes — spawn / wait / pipe / kill / fork over domains

> **Status: PROPOSED — design draft, nothing built.** This is the working tracker for a
> **process abstraction** layered on the machinery that already exists (§14 `Instantiator`
> nesting, §12 fibers/vCPUs, §13 shared regions, `DURABILITY.md` snapshot/clone, the
> `POWERBOX.md` named-capability seam). The forcing function is running a POSIX shell —
> ultimately **Bash** — as a guest; the shell is the best stress test a process layer could
> ask for, and "can Bash run?" is a crisp exit criterion. Like `WASM.md`/`THREADS.md`, fold
> settled parts into `DESIGN.md` and drop this file when the actionable gaps close.
>
> Proposed decision: **D63** (D62 is currently the last). See bottom of file.

The one-sentence design: **a process is a domain plus conventions — no new instructions,
no new isolation machinery.** Spawn is `Instantiator.instantiate_module` with a grant
list; wait is `join`; exec is the `Module` capability; a pipe is a host capability pair;
fork is a durable-domain clone at a quiescent point. Everything lands in the capability
layer (builtin-cap plumbing + embedder-tier host caps + guest libc), touching neither the
verifier nor the confinement lowering.

---

## 0. Orientation — what already exists (verified against the tree)

Grounding, so we don't design what's built:

- **Domains-as-processes exist in embryo.** The §14 `Instantiator` (iface 6) has an 8-op
  surface on all three engines (`svm-interp/src/lib.rs` `CapCall { type_id:
  iface::INSTANTIATOR }`, `svm-jit/src/instantiator_rt.rs`, bytecode via the
  host-orchestrated `VcpuEvent::Instantiate` protocol): `instantiate` (0), `join` (1),
  `spawn_coroutine`/`resume` (2/3), `spawn_demand_coroutine` (4), and the `*_module`
  variants (5/6/7). A child is its own domain: power-of-two sub-window carve, own `Host`
  powerbox and dispatch table, **fuel quota sub-allocated from and capped by the
  parent's**, inherited spawn quota, recursion to any depth.
- **exec is basically built.** A `Module` handle (iface 8) is host-verified code a guest
  may instantiate — and nothing else (`cap.call` on it is inert). Ops 5/6/7 run *that*
  module in a carve that must equal its declared memory; its data segments materialize at
  spawn. A guest can only exec what it was granted.
- **wait exists.** `join` parks **only the calling fiber** (siblings run on) and delivers
  the child's `i64` result or its trap — `waitpid` with an exit status, with the right
  blocking semantics, on the interpreter and bytecode engines.
- **fork's machinery exists, unexposed.** `DURABILITY.md`: freeze → serialize → restore →
  thaw round-trips on both backends **including nested subtrees** (snapshot v4;
  `crates/svm/tests/durable_nesting.rs`), and §10 there says clone "falls out of the same
  machinery at a quiescent point." CoW clone is Phase-4, unlanded.
- **The data plane is free.** A child's carve is a subset of the parent's window (§14
  superset), so the parent seeds argv/env/cwd by ordinary stores before spawn and reads
  results after — no marshalling. Cross-sibling sharing is a `SharedRegion` (§13); one
  grant path into a child's table already exists (`region_grant.rs`).
- **The capability seam is being generalized.** `POWERBOX.md` (in flight): named imports,
  arbitrary host interfaces over `grant_host_fn` (how the `fs` cap works today), and
  `cap.self.resolve` for guest-side discovery by name (`__vm_cap_resolve("fs")`).

The gaps, each named as a "follow-up" in existing code comments or docs:

1. **Children are born destitute.** A spawned child's powerbox holds exactly an
   `Instantiator` + `AddressSpace` over its own window. *"Pass-through of the parent's
   other handles is a follow-up"* (`svm-interp`, `instantiate` op 0). No stdio, no fs —
   no I/O of any kind. **This is the load-bearing gap; everything else stages after it.**
2. **No pipes** — no stream-shaped channel between two domains.
3. **No lifecycle ops** — no non-blocking status poll, no parent-initiated kill (§15
   settles the design: kill rides the §5 detect-and-kill path "via the lifecycle
   capability" — no such op exists), no detach.
4. **JIT children run synchronously at `instantiate`** (`instantiator_rt.rs` header;
   "park only the calling fiber" is its named follow-up). The interpreter enqueues a
   concurrent child vCPU. A shell pipeline needs the async behavior on both backends.
5. **No clone entry point**, and the `fs` cap has no directory ops (no
   stat/readdir/mkdir), which globbing and `cd` need.

---

## 1. Goal & non-goals

**Goal.** A process abstraction sufficient to run a POSIX shell as a guest: spawn a
verified module as a concurrent child with an explicitly granted, attenuated capability
set; wait on it (blocking or polling) for an exit status; kill it; connect processes with
pipes; and — last — fork a running process. Exit criterion, staged (§9): BusyBox
`ash`/`hush` first (designed for fork-less targets), Bash-as-interpreter next, full Bash
(fork-dependent subshells, `$( )`, pipelines of compound commands) as the capstone.

**Non-goals.**
- **No global pid namespace, no process table in the VM.** A child's identity is the
  handle its parent holds; the process tree *is* the grant graph (§15). A shell keeps its
  own small-int "pid" map in guest memory.
- **No same-domain exec-replace.** POSIX `exec` semantics (replace my image, keep my fds)
  are not offered; `fork+exec` collapses to spawn-with-grants, which is what shells
  actually mean. (A shell's rare fd-juggling `exec 3<file` is a guest-libc fd-table
  operation, not a VM one.)
- **No POSIX-complete signals.** v1 signal surface is: parent kills child (uncatchable),
  parent observes child exit/trap. Guest-catchable async signals are parked (§8).
- **No preemptive priorities / scheduling control** beyond the existing fuel/spawn quotas.
- Job-control tty semantics (process groups, `tcsetpgrp`) are out of scope; a shell runs
  non-interactive or line-buffered-interactive.

---

## 2. The model — a process is a domain plus conventions  [PROPOSED]

| POSIX concept | svm realization | status |
|---|---|---|
| process | domain (carve + powerbox + fuel/spawn quota + vCPU) | built (§14) |
| program / binary | `Module` capability (host-verified) | built |
| `posix_spawn` | `instantiate_module` + **grant list** (§3) | grant list missing |
| `waitpid` (blocking) | `join` | built (interp/bytecode); JIT async missing |
| `waitpid(WNOHANG)` / SIGCHLD | `poll` lifecycle op (§4) | missing |
| `kill(pid, SIGKILL)` | `kill` lifecycle op → §5 detect-and-kill | missing |
| exit status | child entry's `i64` return; trap = abnormal termination | built |
| `pipe(2)` | `pipe` host cap: paired stream ends, park-on-empty/full (§6) | missing |
| fd table, `dup2`, redirection | guest libc layer over cap handles (§6) | missing (guest-side only) |
| argv / environ | parent-seeded bytes in the carve, fixed convention (§5) | convention TBD |
| rlimits | fuel quota + spawn quota + carve size — already per-child | built |
| `fork(2)` | durable-domain clone at a quiescent point (§7) | machinery built, no entry point |
| signals | v1: kill + exit observation only (§8) | partially built (host kill path) |

Two structural consequences to state up front, honestly:

- **Carves are power-of-two and subdivide the parent's VA** (§14 honest-bounds). A shell
  running many children needs a buddy-style carve allocator in its runtime layer, and a
  long pipeline occupies several carves at once. Fine at shell scale (a 2^32+ window runs
  hundreds of coreutils-sized children); a design constraint worth remembering for anything
  bigger.
- **Parent sees child memory** (§14 one-way transparency). Processes here are a
  *privilege* hierarchy, not mutually-suspicious peers — correct for a shell and for
  plugin trees; mutually-distrusting siblings need sibling carves (which are already
  mutually invisible) and host-mediated channels only.

---

## 3. Spawn with grants — the load-bearing extension  [PROPOSED]

New `Instantiator` ops (numbers continue the existing 0–7; same scalar-args shape — boring
on purpose):

```
op 8  instantiate_granted(entry, off, size_log2, fuel, grants_ptr, grants_n) -> child | -errno
op 9  instantiate_module_granted(module, entry?, off, size_log2, fuel, grants_ptr, grants_n)
                                                                            -> child | -errno
```

Semantics are exactly ops 0/5 plus a **grant list**: `grants_n` records of 16 bytes at
window-relative `grants_ptr`:

```
{ name_off: u32, name_len: u32, handle: i32, flags: u32 }   // flags reserved-zero in v1
```

For each record, the runtime resolves `handle` in the **parent's** table (a forged/stale
handle fails the whole spawn: `-EBADF`, nothing spawned — fail-closed, no partial child)
and re-grants the same authority into the child's fresh table, registered under `name` so
the child finds it by `cap.self.resolve(name)` — the same discovery convention the `fs`
cap already uses. Out-of-window `grants_ptr`/name ranges are `-EFAULT`; non-UTF-8 or
oversized names `-EINVAL`; a full child table `-EMFILE` (all matching the `fs`-cap errno
conventions).

Design points:

- **Grants are pass-throughs, resolved at grant time** — one hop to the ultimate handler
  at any depth, per §14's "nesting cost is paid at setup." **Attenuation happens at the
  resource capability's own layer**, not in the grant mechanism: to give a child a
  narrower fs, the parent first obtains/derives an fs cap rooted at a subdirectory and
  grants *that*. The grant list stays dumb.
- **Parent-virtualized capabilities** (parent services the child's cap calls — needed
  eventually for a guest that wants to fake an fs for its child) are **deferred**. The
  coroutine `Yielder` (iface 7) already shows the shape; do not build a second mechanism
  until something concrete demands it.
- **Revocation:** v1 grants live as long as the child. Cross-domain revocation (parent
  yanks a grant mid-run) is deferred with the same handle-generation machinery flagged as
  the natural mechanism. Open: what a revoked pass-through returns (`-EBADF` seems right —
  it's how a stale handle already behaves).
- Why scalar args + a window-resident table, not a descriptor struct: matches ops 0–7's
  ABI, keeps the parser trivial to fuzz, and the record format is TLV-extensible via
  `flags` if it ever must be.

---

## 4. Lifecycle — poll / kill / detach  [PROPOSED]

```
op 10 poll(child)   -> 0 running | 1 returned | 2 trapped   (never parks)
op 11 kill(child)   -> 0 | -errno    (idempotent; child's joiner sees trap = Killed)
op 12 detach(child) -> 0 | -errno    (drop our claim; child runs on under its quotas)
```

- `poll` + `join` covers `WNOHANG` and SIGCHLD-driven reaping (a shell's reap loop is
  `poll` over its pid map — no async delivery needed in v1).
- `kill` drives the existing §5 detect-and-kill for **that child's subtree only**. Interp:
  the M:N executor gains per-vCPU-subtree interruption. JIT: today a nested child polls the
  *parent's* interrupt cell (one host interrupt stops the whole tree — the right default);
  per-child kill needs a **per-child cell**, allocated at spawn, checked alongside the
  parent's. This is the one item in this file that touches JIT codegen — still not the
  confinement lowering, but flag it for the same level of review as the epoch machinery it
  extends.
- `kill` of an already-finished child is a no-op success (POSIX-ish); `join` after `kill`
  returns the Killed trap status. Double-`join`/`detach` stay inert faults as today.
- Exit-status convention (guest-level, not VM): shells map `return v` low 8 bits to `$?`,
  trap kinds to `128+n`-style codes. The VM keeps the full `i64` + trap kind; the libc
  layer flattens.

---

## 5. exec, argv, environ  [PROPOSED — convention, zero VM change]

The parent seeds the carve **before** spawn (it sees the superset). Convention (the
"process ABI", owned by the powerbox/libc layer, versioned there):

- A **proc block** at a fixed offset in the child's window (above the module's data
  segments — exact address published by the libc crt, the same way the env-blob `getenv`
  works today): `argc`, `argv` offsets, environ blob, cwd string, umask-ish odds and ends.
- The child's crt (`_start`) reads the proc block and its granted caps
  (`cap.self.resolve("stdin"/"stdout"/"stderr"/"fs"/"proc"/"pipe")`) and builds the usual
  `main(argc, argv)` world. Missing grants degrade exactly as the Lua playground does
  today (`io.open` returns nil when no `fs` was granted).

Note the ordering subtlety for **separate-module** children: their data segments
materialize at spawn and could overwrite a pre-seeded proc block. Resolution: the proc
block address is derived from the module's declared data extent (crt-published), and
seeding happens *after* `instantiate_module_granted` returns for a child spawned
suspended… which v1 children are not. Simplest v1: proc block sits in a fixed-size
reserved region the convention places *above* all data segments (the module's `memory`
declaration already tells the parent where free space starts). Keep it boring; revisit
only if a real module breaks it.

---

## 6. Pipes and the fd table  [PROPOSED]

**v1: a `pipe` host capability** (embedder-tier, exactly like `fs` — `grant_host_fn`, new
iface id from the open range):

```
PIPE_CREATE = 0   create(capacity) -> writes two fresh handles (read end, write end)
                  into the caller's table; returns rd_handle<<32 | wr_handle
PIPE_READ   = 1   read(h, buf, len)  -> n | 0 at EOF | parks when empty & writer live
PIPE_WRITE  = 2   write(h, buf, len) -> n | -EPIPE when no reader | parks when full
PIPE_CLOSE  = 3   close(h) — EOF/EPIPE become observable to the peer
```

- **Blocking = parking the calling fiber**, the same `Blocked`/`Pending` mechanism `join`
  uses — siblings and the peer keep running. This is runtime plumbing (a new park reason),
  not codegen.
- Ends are granted to children via the §3 grant list; a pipeline is: create pipe, spawn
  left child granting the write end as `"stdout"`, spawn right child granting the read end
  as `"stdin"`. The guest-libc **fd table** (int fd → handle + kind) makes `dup2`,
  `2>&1`, and here-docs pure guest-side table edits — zero VM surface.
- `stdin`/`stdout`/`stderr` are just names in the grant list; a pipe end and a §3e
  `Stream` are interchangeable behind the libc `read`/`write`, so "stdout is the terminal"
  vs "stdout is a pipe" is invisible to the child, as it should be.
- **v2 (optional, measured-first):** a zero-crossing guest-library pipe — bounded ring in
  a `SharedRegion` + futex wait/notify. Prerequisite to even consider: verify futex
  wait/notify keys work across *domains* on shared backing (open question O3). Only worth
  it if pipe throughput ever shows up in a real workload; shells are not pipe-bandwidth
  bound.

---

## 7. fork — clone a quiescent child  [PARKED until §9 stage 3]

Full Bash needs `fork` (every subshell, `$( )`, `( )`, `&`, and pipeline element of
compound commands). Everything before it in the ladder (§9) deliberately does not.

Shape when its turn comes:

```
op 13 clone(child, off, size_log2) -> new_child | -errno
```

- **Semantics:** the child (which must be durable-instrumented — the §4 admission rule
  `DURABILITY.md` already enforces for durable subtrees) is driven to a quiescent point
  (the async-STW freeze that already exists), its subtree state captured, and restored
  into a second carve at `off`; both resume. The clone's return-path distinguishability
  (fork's `0` vs child-pid) is a proc-ABI convention: the cloner passes a window address
  whose byte differs in parent vs clone — no VM special case.
- **Handles:** snapshot D-scope already says restore re-grants *authority* and the
  restoring side supplies resources. Here the "restoring embedder" is the runtime acting
  for the parent: v1 policy is **re-grant the same pass-throughs** (POSIX fd inheritance),
  which is the one-liner policy; anything fancier waits for a need.
- **Cost:** v1 is a window copy at clone time (a shell forking a 4 MiB interpreter copies
  4 MiB — fine). CoW rides `DURABILITY.md` Phase 4 (`memfd` + `MAP_PRIVATE`), which this
  file does **not** block on.
- **Constraint to surface early:** `fork` requires the shell module to be built
  durable-instrumented, which costs the §21 instrumentation overhead on may-suspend paths.
  Acceptable for a shell (host-call-bound); measure anyway (R7 in `DURABILITY.md`).

---

## 8. Signals  [PARKED beyond kill]

v1 offers exactly: `kill` (uncatchable, §4) and exit/trap observation (`poll`/`join`).
That covers `SIGKILL`, `SIGCHLD`-as-polling, and — via the host's existing epoch interrupt
— the interactive Ctrl-C story (host kills the foreground child's subtree; the shell,
holding its own handle, survives).

Guest-catchable async signals (`trap` builtin, `SIGTERM` handlers) need a delivery design:
options are a parent-injected fiber, a designated signal futex the guest libc waits on, or
IoRing completion events. **Do not design this yet** — Bash's `trap` is exercised late in
the ladder, and the wrong early choice would ossify. The one hook to keep open: the proc
ABI reserves a "signal word" in the proc block so a cooperative-check convention (guest
polls between commands — which is in fact how shells check traps) can land without a new
mechanism.

---

## 9. The validation ladder — what Bash forces, in order

Each stage is a runnable demo + differential tests; each unlocks the next. (Bash-the-
*language* work — autoconf cross-config, readline-less build, the fs directory ops — rides
alongside and does not block the process layer.)

- **Stage 0 — no processes: Bash/ash as a pure interpreter.** `sh -c 'expansions,
  arithmetic, control flow, functions, builtins'` on the existing Lua-style port model
  (setjmp/longjmp: built; guest malloc: built; varargs printf: built). Needs from this
  file: nothing. Needs elsewhere: **fs dir ops** — `FS_STAT`, `FS_READDIR`, `FS_MKDIR`
  (cwd is guest-libc state; symlinks out of scope) — a small `svm-run/src/fs.rs` protocol
  extension, embedder-tier.
- **Stage 1 — spawn/wait/exec: BusyBox coreutils.** §3 grants + §4 lifecycle + §5 proc
  ABI. Shell runs `ls`, `grep` (BusyBox applets as `Module` grants — the "path lookup" is
  a name→Module-handle table the shell holds; NOFORK applets can additionally run
  in-process, no spawn at all). Prerequisite: **JIT async children** (P0 below).
- **Stage 2 — pipes: `a | b`, redirections.** §6 pipe cap + guest fd table.
  `ls | grep x > out` end-to-end on all three engines.
- **Stage 3 — fork: full Bash.** §7 clone. Subshells, `$( )`, `&`, compound-command
  pipelines. This is the capstone demo — and the point where svm's story visibly exceeds
  wasm's (WASI needed a whole fork of the spec, WASIX, to get here).

BusyBox `ash`/`hush` before Bash is deliberate: they are built for fork-less (NOMMU)
targets, so stages 1–2 deliver a *complete working shell* without waiting for §7, and they
bring coreutils in the same module.

---

## 10. TCB & security posture

Holding the `AGENTS.md` bar — every line is potential TCB, so place each piece:

- **Zero new core surface.** No IR instructions, no verifier change, no confinement-
  lowering change. The §4 masking pass — the security hinge — is untouched by everything
  in this file.
- **Builtin-cap plumbing** (§3 grants, §4 lifecycle, pipe parking): runtime code in
  `svm-interp`/`svm-jit`/bytecode, same tier and same review bar as the existing
  `Instantiator` ops it extends. All of it fail-closed: bad handle / bad range / bad name
  ⇒ negative errno or inert `CapFault`, nothing spawned, nothing granted. The grant-list
  parser gets its own fuzz target (window-relative pointers crossing a hostile guest's
  memory — the same shape as the fs-path fuzzing).
- **The one codegen-adjacent item:** the per-child kill cell (§4) extends the §5 epoch
  machinery. Same review level as the epoch checks; not confinement.
- **Embedder-tier** (pipe cap, fs dir ops, proc-ABI libc, carve allocator, BusyBox/Bash
  ports): +0 escape-TCB by construction, the `svm-run`/guest-code tier.
- **Snapshot-restore** (§7) is the one dependency with real escape-TCB weight — the page
  restore path — and it is already tracked as `DURABILITY.md` R3; fork adds a caller, not
  new mechanism.
- **Authority model unchanged:** a child can only ever hold what its parent explicitly
  granted (D19 attenuation, fail-closed matching); quotas sub-allocate; the grant graph is
  the audit trail (§15). Spectre posture unchanged: in-process children are
  defense-in-depth, distrust still means separate host processes (§2).

---

## 11. Testing plan (repo standard: differential + fuzz, from the first slice)

- Every new op: interp ↔ bytecode ↔ JIT differential (the `jit_instantiator.rs` /
  `bytecode_vcpu_orchestration_instantiate.rs` pattern), including errno paths and trap
  kinds.
- **Grant-list fuzz**: hostile records (OOB pointers, aliasing names, stale/forged
  handles, table exhaustion) ⇒ must fail closed, never partially grant.
- **Pipe determinism**: on the cooperative scheduler (the deterministic oracle, per
  `THREADS.md`), randomized producer/consumer interleavings replay identically; parallel
  backend differentials against it.
- **Kill soundness**: fuzz kill points across a child subtree's lifetime (mid-spawn,
  mid-join, mid-pipe-park, already-dead) — no hangs, no leaks of carve bookkeeping,
  parent's join always resolves.
- **Freeze × processes**: a durable tree frozen while children are parked on pipes/join
  must round-trip (pipes are host resources ⇒ D-scope re-supply; the parked state is
  guest-resident). Extend `durable_nesting.rs`.
- Stage demos become CI gates like the Lua fixtures: BusyBox applet suite (stage 1),
  pipeline byte-exactness vs native shell (stage 2), Bash's own test suite subset
  (stage 3).

---

## 12. Plan tracker

| # | Slice | Depends on | Status |
|---|---|---|---|
| P0 | JIT async children: `instantiate` returns, `join` parks calling fiber (parity with interp) | — | todo |
| P1 | §3 grant list (ops 8/9) + child-side `cap.self.resolve` names, all three engines + fuzz | — | todo |
| P2 | §4 lifecycle: `poll`/`kill`/`detach` (+ per-child kill cell on JIT) | P0 | todo |
| P3 | §6 `pipe` host cap + park-on-empty/full; guest libc fd table + `dup2` | P0 | todo |
| P4 | fs dir ops (`FS_STAT`/`FS_READDIR`/`FS_MKDIR`); cwd in guest libc | — | todo |
| P5 | §5 proc ABI (argv/env/proc block, `_start`, exit-status flattening); carve allocator lib | P1 | todo |
| P6 | BusyBox port; stage-1 demo gate (spawn/wait/exec coreutils) | P1,P2,P5 | todo |
| P7 | Stage-2 demo gate: pipelines + redirections end-to-end | P3,P6 | todo |
| P8 | Bash stage-0 port (interpreter-only; autoconf cross-config, `--noediting`) | P4 | todo |
| P9 | §7 `clone` op over quiescent child (full-copy) + freeze×process tests | P2, durable | todo |
| P10 | Bash stage-3: fork-dependent semantics; suite subset as CI gate | P8,P9 | todo |
| P11 | CoW clone (rides `DURABILITY.md` Phase 4); v2 guest-ring pipes — **only if measured** | P9 | parked |
| P12 | Guest-catchable signals; parent-virtualized caps; revocation | P10 | parked |

---

## 13. Risk register / open questions

| # | Risk / question | Where | Status |
|---|---|---|---|
| O1 | Grant pass-through implementation: child `Host` tables are fresh (`Host::new()`); pass-through needs the resolved entry (Arc'd handler + resource) cloned across tables without exposing host pointers to the guest. Audit the same way `resolve_module` was kept apart from the cap thunk. | §3 | open |
| O2 | Per-child kill on the JIT: per-child cell placement + poll cost vs. today's single parent cell; interaction with fuel/epoch and with a child parked on a pipe. | §4 | open |
| O3 | Do futex wait/notify keys work across domains on `SharedRegion` backing? Gates the (optional) v2 guest-ring pipe only. | §6 | open |
| O4 | Proc-block placement vs. separate-module data segments (§5) — the boring reserved-region convention needs one real module to validate against. | §5 | open |
| O5 | Carve pressure: pipeline of N children = N live power-of-two carves; worst-case fragmentation of the buddy allocator under shell workloads. Measure at stage 2; the answer feeds the window-size guidance, not the VM. | §2 | open |
| O6 | Durable instrumentation overhead on a shell-sized module (fork prerequisite) — R7 of `DURABILITY.md` measured on Bash. | §7 | open |
| O7 | `join` after `detach`, `kill` racing normal exit, grant of a handle that is itself an `Instantiator` (delegated spawning — probably fine and desirable; confirm quota composition). | §3/§4 | open |

---

**[PROPOSED DECISION D63 — processes are a capability-layer composition, not a VM
primitive.]** Spawn/wait/kill/pipe/fork are built by composing the §14 `Instantiator`
(extended with grant-list and lifecycle ops), the §12 fiber-parking runtime, §13 shared
regions, embedder-tier host capabilities (`pipe`, fs dir ops), a guest-libc process ABI,
and (for fork) the D60 durable-domain clone at a quiescent point. **No new instructions,
no verifier or confinement-lowering change**; the only codegen-adjacent work is the
per-child kill cell extending the §5 epoch machinery. Rationale: the process concepts map
one-to-one onto machinery that exists for independent reasons (nesting, durability,
concurrency), so the marginal TCB is capability plumbing; a shell — BusyBox first, Bash as
capstone — is the staged, differential-testable exit criterion, and fork-via-clone is the
distinguishing capability wasm needed a spec fork (WASIX) to approximate.
