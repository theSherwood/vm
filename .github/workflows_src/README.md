# workflows_src — the editable mirror of `.github/workflows/`

The CI token used by agent sessions cannot push files under `.github/workflows/`
(GitHub requires the `workflow` OAuth scope). This directory is the workaround:
agents edit workflow files **here**, and the repo owner applies them by copying
the directory over:

```sh
cp .github/workflows_src/*.yml .github/workflows/
```

(then commit and push with owner credentials). Keep the two in sync in that
direction only — `workflows_src` is the source of truth for *pending* changes;
`.github/workflows/` is what actually runs. After a copy-over the two are
identical until the next agent edit.

## Pending changes not yet copied over

- **`pages.yml` — schedule the deploy instead of per-merge (fixes I75 Pages-deploy starvation).** The
  per-push (`push: [main]`) Pages deploy was starving: under a burst of agent-PR merges each new merge
  supersedes the still-queued deploy (`concurrency: pages`, `cancel-in-progress: false` cancels the
  *queued* older run), so it never wins a runner before the next merge resets it — observed **0 of 30**
  recent Pages runs completed, one sat queued **8 h with zero jobs ever scheduled**, and the live site
  froze days behind `main`. Fix: drop the `push` trigger; deploy on `schedule: */30 * * * *` +
  `workflow_dispatch`. A scheduled run gets a full interval to grab a runner (merge-independent) and
  `cancel-in-progress: false` lets it finish once started. A new tiny **`gate`** job skips the heavy
  build when `main` is unchanged since the last deploy (compares a `DEPLOYED_SHA` marker the assemble
  step now writes to the site root against `HEAD`); a manual dispatch always builds. **On copy-over:**
  merges now publish within ~30 min instead of instantly — use the **Run workflow** button (or the
  `workflow_dispatch` API) for an on-demand deploy. No other job changes. The deeper alternative (fold
  the deploy into the required `browser-real` CI job so it rides a slot that's already scheduled) is
  noted in the session write-up but not done here — it needs `browser-real` to also run
  `build-onramp-assets.mjs` for asset parity, a heavier and harder-to-verify change.

- **I67 apt-source hardening (all `apt-get update` steps)** — every job that runs `apt-get update`
  (the mingw cross lanes, the `clang` reference lanes, and all `llvm-18` install blocks — 10 sites)
  now first `sudo rm -f`s the runner's unused `microsoft*`/`azure*` files under
  `/etc/apt/sources.list.d/`. Those repos are never installed from, but a transient 403/outage from
  their mirror (ISSUES.md I67) kills `apt-get update` with exit 100 before any Rust runs. Removing
  the sources makes the update independent of them. No behavior change on a healthy runner. Pure CI
  infra; no tree code touched.

> **A CI guard now enforces this list.** The `workflows-in-sync` job (`workflows_src == workflows`)
> reds the run whenever any `.github/workflows_src/*.yml` differs from `.github/workflows/*.yml`, so
> pending changes can't be silently forgotten — the run stays red until the owner copies them over
> (`cp .github/workflows_src/*.yml .github/workflows/`) and this section is drained. The guard only
> starts enforcing once it is itself copied into `.github/workflows/ci.yml` for the first time.

- **`cc1-self-compile-giants` job** — a new **nightly** (`schedule` + `workflow_dispatch`) Linux job
  that runs the giant cc1 TUs (`preprocess.c`/`parse.c`/`codegen_ir.c`) through the guest-vs-native
  differential with `SVM_SELFHOST_GIANTS=1`. ~8 min locally (more on CI), too slow for the per-PR gate,
  so it rides the daily cron like `miri`. Together with the five tractable TUs in the always-on
  `cc1-self-compile` job it completes per-TU byte-identity across **all nine** cc1 TUs — the sufficient
  condition for the `chibicc2 == chibicc3` fixpoint. (The always-on job already runs the giant test too
  via `-- --ignored`, but it self-skips fast without the env var.)

- **`full-depth-gates` job** — a new **nightly** (`schedule` + `workflow_dispatch`) Linux job that runs
  the `#[ignore]`d full-depth *correctness* gates that no CI job previously ran: Lua's suite
  (`lua_tlib`/`lua_all`/`lua_sweep`) on both the bytecode engine and the tree-walker, plus the
  whole-language capstones (`demo_tcl_repl_stdin`/`demo_tcl_init_stdin` and the full
  `demo_sqlite_logictest_full` sweep) via `cargo test --test … -- --ignored` from `crates/svm-llvm`
  (workspace-excluded, so run from its dir). Each asserts byte-identity with the native `cc` build.
  `#[ignore]`d only for wall-clock (minutes per suite on the tree-walker), so it rides the daily cron
  like `miri`/giants rather than the per-PR gate — closing the JIT-only blind spot that let the QuickJS
  on-ramp recipe drift unseen once. Capstones self-skip loudly (never fail) without clang/curl/make/
  openlibm, so grep the log for `skipping` before trusting a green run. First green run on CI is the
  real validation of the ~90-min timeout budget.

- **`nim-e2e` job** — builds the real nimony toolchain (`scripts/ci/provision-nimony.sh`, cached) and
  runs `crates/svm-leng/tests/nim_e2e.rs`, which compiles small **Nim source** programs through
  `nimony c` and runs them on both SVM engines. The tests self-skip (pass) in the always-on `check`
  job because the toolchain isn't there; this job provides it so they actually execute. **Two things
  to do on copy-over:** (1) pin `alaviss/setup-nim@0.1.1` by SHA (left as a tag — no vetted SHA to
  hand); (2) confirm the heavy cold build (~10-15 min) fits the runner budget — it's a mirror of
  nim-lang/nimony's own CI and hasn't been run in *this* repo's CI yet, so the first green run is the
  real validation.

*(Previously drained 2026-07-30, when the whole backlog was copied over: the `workflows-in-sync`
guard, nightly-only `miri`, `cross-os` `CARGO_PROFILE_TEST_DEBUG: "0"`, the `playground-assets` job +
the `pages.yml` reachability step, the `bench_chibicc_jit.mjs` / `browser-shell-test.mjs` /
chibicc-asset browser steps, the hardened `embench` fetch, and the full-`fuzz_targets` matrix.)*

> **Reminder for whoever drains this next:** `miri` no longer runs on PRs. If it is still listed as
> a *required* status check in branch protection, remove it there — a skipped required check blocks
> merges.

Remove entries from this list when they land in `.github/workflows/`.
