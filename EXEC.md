# The `exec` capability — one interface, host processes or domains

Design record for the subprocess capability (first consumer: **jacl**,
`theSherwood/jacl_impl/docs/SVM_EXEC_ASK.md` — shell-out for `!cmd` /
`[exec …]`). Settled 2026-07-22 with the §3.6 consumer pinning
(IMPORTS.md §3.6); implementation follows the `svm-fs` mold.

## The one-interface decision

A guest imports **one** interface, `"exec"`, resolved by name
(`__vm_cap_resolve("exec")` + `__vm_host_call`, exactly like `"fs"`).
*What runs* is the wirer's choice, invisible to the guest — the
interposition-invisibility property the import model already guarantees:

| backend | what a spawn is | status |
|---|---|---|
| `host_exec(allowlist)` | a real host OS process, attenuated by an explicit program allowlist (the capability *is* the list, as `host_fs`'s *is* the root) | **BUILT 2026-07-22** (`svm-run/src/exec.rs`) |
| `scripted_exec(table)` | no process at all: a `(argv-prefix → {stdout, stderr, exit})` table — the `mem_fs` analog; what differential tests and wasm/browser embedders grant | **BUILT 2026-07-22** (wasm-safe `svm-exec` crate) |
| `domain_exec` | a **child svm domain** (host-served: the embedder implements the same ops over the Instantiator machinery it already has) | next |
| guest-served | a parent domain serves its child's `"exec"` with **its own code** — the none-the-wiser nested shell | §3.6 `Endpoint` (its first consumer) |

This is how "a shell that manages real host processes" and "the same
shell nested, its parent handling process-like domains, none the wiser"
are the *same program*: the shell's manifest names `"exec"`; the wirer
decides. A shell that *knowingly* wants both at once is not
none-the-wiser — that is a visible, honest choice: two imports (e.g.
`exec.host`, `exec.domain` — dotted names are convention), each wired,
shared, or denied (fail-closed; the guest's fallback runs) by the parent.
Interposition invisibility is per-slot; the manifest is the requirement
statement.

Un-granted, `__vm_cap_resolve("exec")` stays negative and the guest's
fallback runs (jacl: a catchable error value) — a subprocess is pure
authority; without a grant the honest behavior is an error, not
emulation.

## Op protocol (v1 — blocking one-shot)

Ops on the granted handle; errors are negative errno-style returns,
never traps (§3e D42). `argv` is NUL-separated bytes.

```
run(argv_ptr, argv_len, stdin_ptr, stdin_len) -> job | -errno
read_out(job, buf_ptr, buf_cap)               -> n | 0 = EOF | -errno
read_err(job, buf_ptr, buf_cap)               -> n | 0 = EOF | -errno
status(job)                                   -> exit code | -errno
close(job)                                    -> 0 | -errno
```

- **v1 `run` is synchronous-to-completion**: the process runs with the
  given stdin, its output captured; `read_out`/`read_err` then drain the
  captured bytes in chunks; `status` reports the exit code. Blocking
  one-shot is what shell-out needs (the ask, verbatim); pipelines,
  incremental stdin, and signals extend this surface later without
  breaking it.
- **Streaming is reserved, not promised**: the contract permits a future
  backend to return `read_out` bytes *before* the process exits (at
  which point `status` may block). v1 callers that only read after
  `status` are forward-compatible with that; do not write guests that
  *depend* on reads being complete-at-run-return.
- **A program outside the allowlist is a refused op** (negative return),
  not a trap — probeable, like every authority miss. An **empty
  allowlist means "any"** and is a choice the embedder must spell out.

## Contract corners (unified across backends, stated honestly)

- **Exit code domain**: `status` returns an `i64`. `host_exec` reports
  the host's wait status collapsed the POSIX-shell way (exit code
  0–255; signal death reported as `128 + signo`). `domain_exec` reports
  the child domain's entry result verbatim. `scripted_exec` reports the
  table entry. A portable guest treats 0 as success and nonzero as
  failure and nothing more.
- **argv resolution**: `host_exec` matches `argv[0]` against the
  allowlist (the capability's attenuation); `scripted_exec` matches the
  longest argv-*prefix* in its table; `domain_exec` resolves `argv[0]`
  through a Module registry (the `exec_lookup` pattern, personality
  policy).
- **Environment**: not part of v1. A backend that wants env passes it at
  grant time (embedder-side), the way `host_fs` fixes its root — never
  ambient.
- **Blocking profile** (same interface, different resource): `host_exec`'s
  `run` blocks a host thread — it is `Blocking`-shaped and offloadable
  through the existing `IoRing` pool; `domain_exec`'s `run` parks a
  fiber. Guests see one synchronous op either way.
- **Determinism**: `scripted_exec` is fully deterministic (the test and
  browser grant); `domain_exec` is deterministic under fuel;
  `host_exec` is not deterministic and never claimed to be.

## Trust placement

The exec capability is also the trust ladder's bottom rung made
concrete (IMPORTS.md §3.6 consumer pinning): genuinely hostile code
belongs in a **separate OS process** — which is exactly what
`host_exec` grants the ability to create. In-process domains (with
`attest` + attenuated grants) cover trusted and limited-trust code;
tiers 0/1 are never a Spectre boundary (§1a).

## Acceptance (from the ask)

A guest that resolves `"exec"` and runs `echo hi` sees `hi\n` + exit 0
under `host_exec(["echo"])`, byte-identical under a `scripted_exec`
seeded with the same entry, **interp == jit** on both; un-granted,
resolve stays negative and the fallback runs; an allowlist miss is a
negative return, not a trap.

## Implementation shape (the svm-fs mold, exactly)

Protocol constants + the deterministic `scripted_exec` handler live in a
**wasm-safe crate** (`svm-exec`, mirroring `svm-fs`) so the browser
cdylib can grant it; the real `host_exec` backend and the `HostCap`
constructors live in `svm-run` (`exec.rs`, mirroring `fs.rs`).
`domain_exec` lands beside them when built; the guest-served backend is
§3.6/Endpoint work and is recorded there.

**As built (2026-07-22).** `svm-exec` holds the protocol (op codes, errno,
NUL-argv parsing) plus the shared `JobTable` — every backend routes its
non-`run` ops through the one table, so read/status/close semantics are
backend-identical by construction — and `scripted_exec_handler` (longest
argv-prefix wins; a miss is `-EPERM`, the *same* refusal an allowlist miss
produces, so the failure mode doesn't reveal the backend either).
`svm-run/src/exec.rs` adds `host_exec(allowlist)` (blocking one-shot spawn,
stdin fed then closed, both streams captured, POSIX-shell status collapse)
and the `HostCap` constructors, re-exporting `svm-exec` as one surface.
Acceptance pinned by `crates/svm-run/tests/exec_cap.rs`: `echo hi` →
`hi\n` + exit 0, byte-identical host vs scripted, on all three backends;
un-granted resolve negative with the fallback running; allowlist miss
probeable, never a trap (the host test is unix-only — Windows has no
`echo` executable; the protocol is covered everywhere by the scripted
differential).
