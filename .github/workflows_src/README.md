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

- **ci.yml** (2026-07-23):
  - `check` job: `cargo test --workspace -j 2` — the ISSUES.md **I30** durable
    mitigation (three sightings; bounds the parallel-link memory peak that OOMs
    the runner agent).
  - `check`, `real-browser`, and `fiber-scaling (stack-check + arena-stacks)`
    jobs: `timeout-minutes` on the network-fetch steps (apt mingw ×2, Playwright
    install) — the **I34** hardening (a wedged mirror fails fast into a re-run
    instead of pinning a runner; both stall modes were observed 2026-07-23).

Remove entries from this list when they land in `.github/workflows/`.
