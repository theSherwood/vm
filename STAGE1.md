# Stage 1 — external commands: `fork`/`exec`/`wait` for the shell

Stage 0 (PROCESS.md §10 / S7) proved a real command interpreter runs on the
`svm-posix` personality: redirection, pipelines, lists, variables, globbing,
`if`/`then`/`else`, and a dozen builtins, all differential-tested interp==JIT
(`crates/svm/tests/c_shell.rs`). Everything there is **in-process** — pipelines
stage through memfs temp files, and there are no child processes.

Stage 1 gives the shell **external commands**: run a program that is *not* a
builtin as its own confined child domain, deliver it `argv`, inherit stdio, and
collect its exit status into `$?`. This is `posix_spawn` + `wait` — sequential,
no fork-returns-twice yet.

## The substrate already exists

The child-domain machinery is built and CI-gated; Stage 1 is **personality
glue, not new substrate**:

- **`Instantiator` (iface 6)** — `instantiate_module` (op 5) spawns a child
  running a **separate host-verified `Module`** confined to a carve of the
  parent's window; `join` (op 1) parks the caller until the child completes and
  yields its result. Proven on both backends (`separate_module.rs`,
  `jit_separate_module.rs`), including data-segment materialization and
  confinement (a child cannot touch bytes outside its carve).
- **`Module` (iface 8)** — host-verified code a guest may instantiate. The host
  grants a `Module` capability (`Host::grant_module`); on the JIT the child is
  resolved through `svm_run::module_resolver` (never guest-reachable) and
  compiled **at instantiate** — §14's "nesting cost paid at setup".
- **Grants into children** — `instantiate_granted` (op 8) / `instantiate_named`
  (op 11) re-grant the parent's own capabilities (stdout/stderr/stdin) into the
  child, discovered by name via `cap.self.resolve` (`instantiate_granted.rs`,
  `instantiate_named.rs`). This is stdio inheritance.
- **argv delivery** — a child runs over the **parent's shared window backing**;
  the carve is not zeroed, so the parent seeds `argv` bytes into the child's
  carve before `instantiate` and the child reads them at low offsets. (The
  child's own data segments materialize over the carve at spawn, so `argv` goes
  where no segment lands.) Proven by `stage1_spawn_wait.rs`.
- **exit status** — `join`'s `i64` result is the child's return value; the
  personality maps it to POSIX's 8-bit `$?` convention.

## The mapping — BusyBox-multicall in miniature

A shell's `exec` and BusyBox's applet dispatch are the same shape: a program
image that, run with a different `argv[0]`, *is* a different command. In this
substrate an "external program" is a verified `Module`; running one is
`instantiate_module` + `join`. The shell's `PATH` becomes a **name → `Module`
map** the personality holds; command lookup is a map lookup; `exec` is spawn.

## Slice plan

0. **Unmodified `main(argc, argv)` on chibicc** *(done —
   `stage1_argv_main.rs`)* — the "as close to native as the security model
   allows" milestone. An **ordinary** C program — `int main(int argc, char
   **argv)`, `write(1, …)` to an *ambient* fd, no capability threading in the
   source — compiles through chibicc and runs with a real `argv`. chibicc's
   synthesized `_start` now parses the §3e powerbox args buffer at
   `POWERBOX_ARGS_BASE` (`{argc, envc}` + packed strings) into an `argv[]`
   pointer array parked at the entry SP, then calls `main(main_sp, argc, argv)`
   with `main`'s frame relocated a page above; writable globals shift past
   `POWERBOX_ARGS_END` so the seeded argv never collides with a global (both
   opt-in for a `main`-with-params, so every `main(void)` program — incl. the
   Stage-0 shell — is byte-identical). Modeled on svm-llvm's `synth_start_argv`
   (already vs-native there, but that frontend is an excluded LLVM-dependent
   crate; the self-contained demo needs chibicc parity). Output tracks argv,
   exit code = `argc`, differential interp==JIT; no confinement-path change. This
   is the crt that makes a compiled C program a **first-class bash command**.
1. **Spawn/wait spike** *(done — `stage1_spawn_wait.rs`)* — a differential
   (interp+JIT) test proving the core: a parent seeds `argv` bytes into a child's
   carve, `instantiate_module`s it, `join`s, and the child's return (a function
   of the seeded bytes) is the parent's result — with the child's output also
   readable from the shared carve. No shell yet; de-risks the mechanism.
   - **argv[] vector ABI** *(done — `stage1_argv_vector.rs`)* — pins the real
     `main(argc, argv)` marshalling: `argc` (i32), an `argv[]` pointer array, and
     a string blob laid in the child's carve. An applet reads `argc`, follows
     `argv[1]`'s pointer to its bytes, and echoes them to the granted stdout —
     proving pointer-array *indirection* (the output tracks `argv[1]`, not a flat
     read), status = `argc`, differential interp==JIT. This is the layout the
     personality's `spawn` will lay down.
2. **stdio-inherited child** *(done — `stage1_stdio_child.rs`)* — a same-module
   BusyBox-applet child inherits a granted `stdout` (`instantiate_named`, op 11)
   and echoes its parent-seeded `argv` to it: a real external `echo` — argv in,
   bytes out through inherited stdio, status back — differential interp==JIT.
   - **Foreign-program variant** *(done — `stage1_foreign_command.rs`)* — the
     general `exec` case: the command is a *separate* verified `Module` (a
     distinct binary), spawned via `instantiate_module` (op 5). Separate-module
     children have no stdio-grant op, so the shell uses the **parent-as-pager**
     model — the child writes output into its carve and the parent forwards
     those bytes (length = the child's `join` return) to its own stdout.
     Differential interp==JIT.
3. **multi-applet dispatch guarantee** *(done — `stage1_applet_dispatch.rs`)* —
   one binary carries several applets (`true`→0, `false`→1, `echo`→writes+3),
   and spawning a chosen entry yields that applet's own `(stdout, status)`. This
   is the substrate guarantee the shell's command dispatch rests on: look a
   command up, spawn its entry, thread its exit code into `$?`. Differential
   interp==JIT. The name→entry map itself is trivial glue and lands in slice 4.
   - **C-applet ABI** *(done — `stage1_granted_argv_applet.rs`)* — the applet
     receives its `stdout` as an *entry argument* (via `instantiate_granted`,
     op 8 — the handle is the child's 3rd arg) and writes through it, rather than
     resolving by name. This is the shape a **chibicc-compiled** applet must take:
     the frontend's generic capability import passes the handle as the *first C
     argument at runtime* (`codegen_ir.c` §7) and cannot emit `cap.self.resolve`,
     so `applet(inst, addrspace, stdout_h)` writing through `stdout_h` is the
     natural form. Proven with a seeded-argv echo, differential interp==JIT.
4. **exec a compiled-C command (pure status)** *(done —
   `stage1_exec_command.rs`)* — a "shell" parent spawns a *separate*, unmodified
   `int main(int argc, char **argv)` C program via `instantiate_module` (op 5),
   delivers `argv` through the §3e args buffer seeded into the child's carve, and
   `join`s for `main`'s return — the value a shell records in `$?`. The one
   enabler was a chibicc **`--child-entry`** flag: it emits function 0 with the
   §14 child ABI (`(i64 starter) -> (i64 status)`, `main`'s int widened) instead
   of the paramless top-level powerbox entry, so the program is spawnable while
   still parsing the args buffer into `main(argc, argv)`. Status tracks argv
   (real delivery), differential interp==JIT; **no new substrate op** (rides
   op 5 + op 1). Empirically found: an unmodified powerbox `_start` (`() -> i32`)
   ThreadFaults under `instantiate_module` — the child-entry signature is the fix.

4b. **exec a compiled-C command with inherited stdout** *(done —
   `stage1_exec_stdout.rs`)* — the full external `echo`. This closes slice 4's
   gap with a new substrate op, **`instantiate_module_named` (Instantiator
   op 13)**: the union of `instantiate_module` (op 5 — resolve + compile a
   host-granted `Module`, materialize its data into the carve) and
   `instantiate_named` (op 11 — re-grant caps into the child's powerbox by name).
   It is the only op that runs a foreign program **and** hands it capabilities,
   so a compiled command resolves an inherited `stdout` by name and `write(1, …)`
   lands in the shell's sink. The shell parent re-grants its own `stdout` under
   the name `"stdout"`; the command's `--child-entry` `_start` resolves it. Built
   on **both backends** (interp dispatch decode + JIT `instantiate_module_named`
   thunk, the JIT given both the module resolver and the named-grant hooks),
   differential interp==JIT, output tracking argv. **Authority-TCB, not
   escape-TCB (§2a): the D63/D38 carve masking is untouched** — op 13 is literally
   the union of two existing, fuzzed decode paths through the same confined-child
   spawn. Existing instantiate suites unchanged.

4c. **op 13 driven by a compiled-C shell** *(done — `crates/svm/tests/c_shell_exec.rs`)*
   — where 4b's parent was hand-written IR, here a **chibicc shell** (`main(argc,
   argv)`) parses its own powerbox args, looks the command up via a host fn,
   seeds the command's `argv` into a 128 KiB carve, lays a `"stdout"` grant record,
   and drives `instantiate_module_named` (op 13) + `join` through capability
   imports (`Resolved::CapBound`, the `Instantiator` baked in) — the whole
   external-command path emitted by the frontend, differential interp==JIT.
   This surfaced a **latent JIT/interp differential gap** (now fixed, no
   confinement code touched): the JIT's `lower_instantiator` demanded an
   *exact-width* Instantiator contract (i32 child handle / i32 `join` arg), but
   chibicc widens every scalar to an i64 slot (`int __spawn(...)` → `… -> (i64)`),
   so **no compiled-C program could drive the Instantiator on the JIT** — every
   op-13/op-1/op-5/op-8/op-11 call fell to a `CapFault` the interpreter never
   raised. The interpreter already tolerated the width (it reads args as i64 slots
   and coerces each result to the declared type, `slot_to_val`); the JIT now
   mirrors that with `slot_i64`/`slot_i32`/`result_as` coercions and a
   width-tolerant shape gate (scalar-int, arg-count) across every Instantiator arm.
   A non-scalar shape, too-few args, or unknown op still lowers to a runtime
   `CapFault` (never a compile-time rejection). Guarded portably by the
   `wide` arm of `stage1_exec_stdout.rs` (i64-declared op-13 result + join arg,
   hand-IR, no chibicc). *Follow-up:* folding this into the full `c_shell.rs`
   builtin dispatch (its personality-heap-at-`win/2` layout vs. a 128 KiB command
   carve).

### Power 2 — the endpoint direction (deferred, S9)

Op 13 is **forwarding** (capability model "Power 1"): a parent hands a child a
capability it already holds. That covers `cmd` writing straight to the terminal.
It does **not** cover the shell *intercepting* a command's output — `cmd > file`,
`cmd | other`, capturing into a shell buffer — because there the shell must not
forward the real `stdout` but **serve** the child's stdout with its own code
(capturing bytes, feeding a pipe). That is capability-model **Power 2**: a guest
minting a capability whose implementation *is* its own code, so a child's
`cap.call` on it parks and wakes the parent (§14 "the parent's own handler /
pay-for-what-you-virtualize"). The primitive is the **`Endpoint`** (PROCESS.md
§4, `[PROPOSED]`; S9 on the roadmap): `mint(sig) -> (serve_end, client_template)`,
`serve`/`reply`. A guest-served endpoint is what makes a parent a **personality /
kernel for its children** (parent-as-POSIX-kernel, parent-as-pager) — the
keystone of self-similarity. It is **not built**. Until it lands:

- **`cmd` → terminal**: op 13, forwarding the real stdout. **Done.**
- **`cmd > file`, `cmd | cmd2`** (shell-side interception): needs `Endpoint`
  (Power 2). The stopgap is the parent-as-pager model (`stage1_foreign_command.rs`
  — the command writes to its carve, the parent forwards), which requires a
  command written to output to memory rather than an ambient fd, so it is **not**
  a drop-in for unmodified compiled commands. Real redirection/pipelines of
  external commands wait on the endpoint work.

5. **`spawn` in the personality** *(substrate done — `crates/svm/tests/stage1_posix_spawn.rs`)*
   — `svm-posix` gained a minimal **`exec` surface**: a `PATH` registry (`name →
   Module` handle, `Posix::register_command`) reached by an `exec_lookup` op, plus
   an `exec_stdout` op handing back the forwardable stdout `Stream`. A compiled-C
   shell **running on the real personality** now dispatches an unknown command to
   a spawned external child instead of `<cmd>: not found`: it reads its own `argv`
   (`argc`/`argv`), looks the command up, and — the spawn being the shell's own
   `Instantiator` op 13 + `join` (`Resolved::CapBound`) — re-grants `stdout` by
   name and threads the child's `argc` into the status. The **two stdout models
   are unified**: `Posix::set_stdout_sink(host.shared_stdout())` routes the
   personality's fd-1 writes and the child's re-granted `Stream` writes into one
   `Host` sink, so shell and command output interleave. Differential interp==JIT,
   three paths (builtin / external / not-found). Confinement untouched (op 13 is
   the existing fuzzed spawn path; the personality is authority-TCB, §2a).

   **Folded into the full Stage-0 shell** *(done — `crates/svm/tests/c_shell.rs`)*:
   the real `c_shell.rs` shell now spawns external commands from its command
   dispatch — the `else` (was `<cmd>: not found`) branch does `exec_lookup` and,
   on a hit, `spawn_cmd` (grant record + args carve + op 13 + `join`), threading
   the child's status into `$?`. The layout tension is resolved as the focused
   test does: a 384 KiB `pool` static forces a window with a 128 KiB-aligned
   command carve **below the stack**, and the personality heap moves to the top of
   the window (the shell never `malloc`s, so it is never touched). `run_shell`
   takes an optional PATH of `(name, C source)` commands; with none registered
   `exec_lookup` always misses, so the 24 existing shell tests are unchanged. Two
   new tests cover a spawned command's argv delivery + `$?` and its status flowing
   through `&&`/`||`. A `>`/`|` redirect on an *external* command is not honored
   (the command always writes to the terminal sink) — that is the Power-2
   `Endpoint` gap below, not a regression.
6. **Pipelines across real children** — replace the memfs-temp pipeline staging
   with concurrent OS-thread children communicating through a granted
   `SharedRegion` + canonical-key futex (PROCESS.md §4 "revised async-children
   plan"). This is the jump from sequential spawn/wait to true concurrency.
   **[PROMOTED 2026-07-22 — an svm-owned todo, consumer-pinned.]** jacl (the
   first shell-like language targeting svm) needs concurrent stages soon after
   sequential; this does not wait for a further request. The remaining build is
   step 2 of the revised plan (OS-thread children in own guarded windows; the
   canonical-key futex, step 1, landed as S1b) — sequential-first, concurrency
   promptly after.
7. **`fork`/`clone`** — the parked-domain clone path (PROCESS.md §7), the last
   piece for shells that fork *themselves*.

Security posture is unchanged: children keep their **own guarded windows**; the
D38 confinement lowering (the most sensitive code in the tree) is not touched.
Stage 1 only *composes* existing, fuzzed primitives.

## Known caveat — crash handling waits for async convergence

A crashing command must not crash the shell, which needs `poll` (op 9:
`0` running / `1` returned / `2` trapped) to detect a trapped child and
`detach` instead of `join` (a `join` propagates the child's trap to the
parent). But **`poll` after a synchronous spawn is not yet backend-portable**:
the interpreter runs a child lazily (at `join`), so `poll` reports `0`
(running); the JIT runs it eagerly on its own OS thread, so `poll` reports `1`
(returned). A differential `poll`-based control flow therefore disagrees today.
This converges with the async-children work (slice 5 / PROCESS.md §4), which is
where crash-status mapping (`$?` = 128 + signal) lands — not before.
