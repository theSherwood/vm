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

- **ci.yml** (2026-07-24): `check` job gains `env: CARGO_PROFILE_TEST_DEBUG: "0"` — the I30
  linker-OOM runner deaths recurred twice on PR #427 *with* the `-j 2` cap (sightings 4-5);
  dropping test-profile debug info removes the dominant per-link memory term. See ISSUES.md I30.
- **ci.yml** (2026-07-24): `embench differential` fetch hardened with `curl -f --retry 5
  --retry-all-errors` — codeload occasionally serves an HTML error page that `tar xz` can't
  detect ("not in gzip format"). See ISSUES.md I18 class 4.
- **ci.yml** (2026-07-24): `fuzz` matrix expanded from the 6 escape-TCB targets to **every**
  target in `fuzz/fuzz_targets/` (adds `onramp_diff`, `roundtrip`, `opt_sccp`,
  `opt_ssa_roundtrip`, `coverage_walk`, and the `durable*` freeze/thaw family) — no
  built-but-unwired fuzzer. Job renamed `cargo-fuzz (all targets)`.
- **ci.yml** (2026-07-24): `cross-os` job — removed the stale commented-out `continue-on-error`
  TODO; the job is already gating.

Remove entries from this list when they land in `.github/workflows/`.
