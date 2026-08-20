# bash — GNU bash on the LLVM on-ramp (#802)

The bring-up of the umbrella's (#794) target program: **the literal GNU bash source**, compiled
through the on-ramp as one whole-program module, hosted by the svm-posix personality this tree
built for it (fork #863, signals #796, job control #798, pipes #972, exec + `/bin` #801, the
controlling terminal #797, `longjmp`-from-a-handler #802-slice-1).

## Slice 2 (DONE) — translate + verify

`./build_bitcode.sh`: fetch bash 5.2.21 (fetched-not-vendored, GPLv3) → configure the bring-up
config → **native oracle build** (also generates y.tab.c and the `.def`-built builtins) → per-TU
bitcode with each Makefile's own flags (152 TUs: the link-line objects + libbuiltins/libglob/
libsh/libhistory(hist* only — the rest are readline standalone shims that duplicate bash's own)/
libtilde) → `llvm-link` + `bash_shim.c` + the reused waist → **translate (~0.8 s), verify:
clean**. Gate: `demo_bash_translates_and_verifies` (svm-llvm `translate.rs`, `#[ignore]`d
for wall-clock).

The bring-up config (`configure` flags, each with a reason in the script): `--without-bash-malloc`
(the waist malloc, not sbrk), `--disable-readline` (non-interactive first; interactive rides the
#797 terminal in slice 4), `--disable-nls`, `--disable-net-redirections` (no sockets), and
`ac_cv_type_long_double=no` (the printf builtin's `%Lf` would need x86_fp80 — denying the type
keeps `floatmax_t = double` in guest AND oracle). **Job control stays on.**

## Slice 3 (DONE) — first run: `bash -c` differential vs the oracle

The OS lane: the embedder grants the svm-posix personality as the **named "posix" capability**
(`svm_run::posix::posix_cap` + `run_with_caps` — the `posix_cap.rs` idiom); the shim's **band 0**
resolves it once (`__vm_cap_resolve("posix")`) and defines the real libc entry points
(open/read/write/stat/dirs/signals/termios/process ops) over `__vm_host_call` op dispatch,
marshaling C conventions (NUL strings, glibc struct layouts) to the op ABI. No translator lane was
added — this is the same named-cap route `posix_cap.rs`/`fs_cap.rs` already prove. bash runs the
**interpreter** (setjmp/fork are interp-only tiers). Gate: the run half of
`demo_bash_translates_and_verifies` — five `bash -c` scripts (echo/vars/arithmetic/for/functions),
stdout + exit byte-compared against the native oracle under identical argv/env.

## The gap-walk (the Tcl discipline: every gap gets a pinned unit test)

1. **`align 4294967296`** — clang stamps the max alignment (2^32, one past `u32`) on
   deliberately-trapping null stores (bash's `programming_error`). The `.ll` parser now saturates
   an alignment literal instead of refusing the module. Pin: `align_u32_max_saturates`.
2. **Old-C call-site drift** — bash's empty-parens prototypes (`extern void f ();`) let call
   sites invent their own function types: `add_unwind_protect(fn, 0)` is typed
   `(ptr, i32, ...)` at the site against a plain `(ptr, ptr)` definition. The native ABI hides
   the drift; the lowering now follows the **definition** for direct calls — arity split,
   va-area deposit only for a genuinely variadic callee, and integer args coerced to the
   definition's widths. Pin: `old_c_call_site_drift_follows_the_definition`.
3. **Old-C INDIRECT call-site drift** — the function-pointer twin: `typedef int Function ()`
   tables call cleanups through `(ptr, ...)` sites whose runtime target is a plain `void ()`
   (`add_unwind_protect(pop_stream, NULL)` → `(*cleanup)(arg)`). The strict typed `call_indirect`
   trapped `IndirectCallType` (a pinned security contract — never loosened). The translator now
   routes a **varargs indirect site** (the old-C unspecified-params marker; ANSI code never has
   one) through a synthesized **static dispatcher**: a funcref-equality chain direct-calls each
   address-taken candidate with its definition's own signature (args width-coerced, missing result
   padded 0), and everything else — exact-typed targets, real variadic targets, unknown funcrefs —
   falls to the strict `call_indirect` unchanged. CFI is never widened (every arm is a direct call
   to a statically-named function). Pin: `old_c_indirect_call_drift_dispatches_to_the_definition`.

Shim-side stub walk to first run: `qsort` (the tcl_shim heapsort — the on-ramp does not
synthesize it), `strerror` (glibc-matching strings — bash prints them in error messages the
differential compares), `strnlen`/`strdup`/`strncpy`/`strcat`/`strstr`/`strcasestr`/`strchrnul`,
`imaxdiv`, and the wide-char band (`mbsrtowcs`/`wcs*`/`isw*`/`towlower`/`wctype` — ASCII,
MB_CUR_MAX = 1).

## Slice 4 (fork/pipes rung DONE) — command substitution, subshells, builtin pipelines

bash **forks** on this lane: `` echo `echo nested` ``, `$(…)` (multiple, multi-line), `(subshell)`,
and builtin pipelines (`echo a | { read x; …; } | …` with trailing commands) all byte-match the
oracle. What it took (each pinned):

1. **The core-pipe builtins on the on-ramp** — `__vm_pipe`/`__vm_read`/`__vm_write`/`__vm_close`
   now lower exactly as the chibicc frontend's (`codegen_ir.c`), so band 0's #972 tag-protocol
   wrappers complete (interpreter tier — `CAP_SELF_PIPE` needs the `Real` scheduler). Pin:
   `core_pipe_builtins_roundtrip`.
2. **`posix_cap` grants everything `grant` grants** — the async-signal door (which carries the
   #799 caller-request door: without it `fork` was `-ENOSYS`), the #972 exec-remap hook, the #801
   op vtable (svm-run `posix.rs` + the new `svm_posix::cap_signal_source`/`cap_exec_remap_hook`/
   `cap_vtable`).
3. **A pristine guest-JIT grant forks** — the fixed powerbox prefix mints an (unused) `Jit`
   domain on every svm-run host; `fork_powerbox` refused it wholesale. A domain with no units, no
   installs, no native ctx now duplicates as an equally-empty grant (same quota, same index);
   live JIT state still fails closed.
4. **`fork_private` copies the `vm_map`-committed tail pages** — the waist malloc's heap lives in
   the reserved tail, so every malloc-using on-ramp program's fork was refused (`-EAGAIN`, which
   bash surfaces as `fork: retry`). Plain `Rw`/`Ro` tail pages now copy page-wise into the twin;
   `Backed` (§13) stays fail-closed. Pin: `fork_private_copies_vm_mapped_tail_pages`.
5. **The any-child blocking wait benches** — bash's no-job-control `waitchld` blocks in
   `waitpid(-1, …, 0)`; the #799 bench covered only specific pids, so the parent raced past its
   unfinished subshell/pipeline (the `end` before `in-5`'s newline diff). Op 28 now benches on
   the lowest live core-twin child for `-1` (correct for foreground waits — the caller loops
   reaping until all children are gone; the true any-child park key is a later rung).

End-to-end pin: `fork_over_the_named_posix_cap_copies_the_heap` + five fork-era scripts in the
capstone differential.

## Slice 4 (exec rung DONE) — external commands: the #801 `/bin` from bash

bash **execs**: `/bin/echo`, PATH lookup, exec'd pipelines (`seq 3 | sort | uniq | wc -l` — four
exec'd programs), redirections to memfs files, and command substitution over exec'd stages all
byte-match the oracle. The lane: `posix_libc/exec.c` (the #801 execve/execv/execvp, staged-pack
argv over the args region + `CAP_SELF_EXEC` image-replace) links as guest code — its `__px_*`
externs bridge in band 5, and `__vm_exec_module` joins the on-ramp's core-builtin lowerings
(self-namespace op 14, mirroring chibicc). `stage_bin.sh` compiles the `posix_utils` coreutils
(the chibicc world, unchanged) to `.svm` command modules; the harness grants each as a `Module`
and registers it as a filesystem executable inside the posix grant (`c_posix.rs`'s
`stage_executable` shape — `bash_probe` takes `BASH_PROBE_BIN=<dir>`). Gate: eight
external-command scripts in the capstone differential (18 scripts total).

## Slice 4 (signals rung DONE) — traps deliver

`trap … INT` + `kill -INT $$` runs the trap (in the parent AND in a fork-twin subshell), ignored
(`trap "" INT`) and repeated deliveries match, EXIT traps compose with subshells. The whole fix
was **one shim line**: async delivery (#796 L2) is gated on a registered handler stack (the
interp's safepoint redirect runs the C handler on a dedicated stack) and bash never calls
`sigaltstack` on this config — band 0 now registers a static 16 KiB stack in a ctor
(`llvm.global_ctors`, which the synthesized `_start` already runs). Gate: five trap scripts in
the capstone differential (23 scripts total).

Known nuance (deferred until a real script trips it): `(kill -INT $$); echo rc=$?` — `$?` after
the shell ITSELF is signaled from a subshell while waiting differs (svm 128, native 0: bash's
`wait_sigint` discard logic vs the personality's `128+sig` zombie status encoding).

## Interactive rung 1 (DONE) — `bash -i` on the #797 controlling terminal, foreground

The harness lane: `svm_run::posix::posix_cap_terminal` enables the #797 terminal at grant time;
the embedder types with `Posix::feed_terminal` from a feeder thread while the shell runs (the
`run_interp_terminal` witness shape). What works, session-proven and gated (the interactive block
in `demo_bash_translates_and_verifies`; `bash_probe` drives ad-hoc sessions via
`BASH_PROBE_TERM='line;^C;line;^D'`):

- **The prompt loop** — bash comes up `flags=himBHs` (interactive AND monitor/job-control mode —
  richer than native bash under a pipe, which loses `m`), prints PS1 to fd 2 between commands,
  reads canonical lines through the feed-time discipline (echo on the captured stdout).
- **`^C` at the prompt** — VINTR through the discipline → SIGINT to the foreground group → bash
  aborts the line, `$? = 130`, fresh prompt (native-exact).
- **`^D` on an empty line** — true EOF → bash prints its `exit` farewell and exits with the last
  command's status.
- **Commands at the prompt** — builtins, external commands, and pipelines (`seq 3 | cat`) run
  exactly as in `-c` mode.
- Shim fix en route: `getcwd(NULL, 0)` (the glibc allocate-extension — bash's shell-init cwd
  probe) now allocates instead of failing into the `shell-init: error retrieving current
  directory` warning.

## What remains (the slice ladder from the #802 sketch)

- **Interactive rung 2 — background jobs**: `^Z`/`jobs`/`fg`/`bg`. The walk found the gap: an
  exec'd child's fd 0 is a **drained stdin snapshot**, not the terminal — `cat` at the prompt
  EOFs instantly instead of reading the terminal, so there is never a live foreground job to
  stop. Needs the #801 exec plumbing to hand the child the terminal-backed fd 0 (and then the
  VSUSP feed path + the stop report through interactive `waitpid(WUNTRACED)` — the #798
  machinery, already personality-side).
- **Slice 4 remainder**: here-docs, the `$?` edge above.
- Known band-0 papering (revisit when a differential trips over one): `fstat` synthesizes a
  chr-device for fds 0-2 and re-stats the recorded open path otherwise; `st_ino` is a path hash
  (same-file checks distinguish paths, not hardlinks); `sigsuspend` returns `EINTR` without
  suspending; readline/progcomp externs stay trap stubs (`--disable-readline`; the `complete`
  builtin would hit them).

| File | Role |
|---|---|
| `build_bitcode.sh` | the faithful fetch→configure→oracle→bitcode→link→translate pipeline |
| `bash_shim.c` | the bash-specific libc/OS surface (grows per slice; see its header) |
