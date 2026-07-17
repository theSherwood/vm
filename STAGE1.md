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

### Known gap — stdout-inheriting commands need a module+grant primitive

Slice 4 handles **no-capability** commands (their `_start` resolves nothing).
A command that writes to `stdout` needs its `stdout` cap **re-granted into the
child**, but the grant ops (`instantiate_granted`/`_named`, ops 8/11) are
**same-module** only — there is no `instantiate_module`+grant op, and a raw
handle can't be seeded into a child's cap table (re-granting must insert it).
So a stdout-inheriting compiled-C command is blocked on a new substrate
primitive: **spawn a verified `Module` as a confined powerbox child** (re-grant
named caps + seed the args buffer + invoke its entry). That op — authority-TCB,
not escape-TCB (§2a); the carve masking is untouched — is the next real
substrate slice. Until it lands, output can only be forwarded parent-as-pager
(the `stage1_foreign_command.rs` model, which needs the command to write to its
carve rather than an ambient fd).

5. **`spawn` in the personality** — give `svm-posix` a `PATH` registry of command
   `Module`s and the `Instantiator` handle, so the Stage-0 shell dispatches an
   unknown command to a spawned child instead of `<cmd>: not found`, threading
   the child's status into `$?`. The compiled shell drives `instantiate_module`/
   `join` through generic capability imports (the handle-as-first-arg convention,
   `codegen_ir.c` §7); commands are `--child-entry` modules the embedder mounts.
   Full stdout inheritance rides the module+grant primitive above.
6. **Pipelines across real children** — replace the memfs-temp pipeline staging
   with concurrent OS-thread children communicating through a granted
   `SharedRegion` + canonical-key futex (PROCESS.md §4 "revised async-children
   plan"). This is the jump from sequential spawn/wait to true concurrency.
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
