# Active todo tracker

The single index of open work. Each row points at the document that owns the
design — this file tracks *that the item exists and where it stands*, nothing
more (keep it short; keep it current). Items land here when they are deferred
with rationale, and leave when a BUILT block lands in the owning doc.

## Consumer-critical (jacl path)

| item | owner | status |
|---|---|---|
| **Concurrent stages** — pipelines across concurrently-running children over `SharedRegion` + canonical-key futex (STAGE1.md item 6) | STAGE1.md, PROCESS.md §4 revised async plan | **substrate BUILT 2026-07-23** (cross-domain canonical keys + region regrant + the ring-pipeline pin, interp); remaining below |
| ↳ JIT pipeline — aliasing a `SharedRegion` into a *separate child window* | STAGE1.md item 6 BUILT block | **BUILT 2026-07-23** — async granted children (ops 8/11/13 on OS threads) + child wait/notify over the parent domain's shared futex; the aliasing PAL already existed (`MprotectWindow::map_region` works against any `GuestWindow`) |
| ↳ Personality `\|` wiring — `c_shell` pipelines over region rings instead of memfs temp files | STAGE1.md item 6 BUILT block | **BUILT 2026-07-23** — pure-filter stages run as concurrent `__stage` children over ring futexes (fallback to temps otherwise); redirects on *external* commands remain Power-2-gated (Endpoint) |
| **Endpoint self-mint** — a domain minting a live offer over its *own* impl-export (the distrust-parent variant; parent-mediated sibling-as-service is BUILT) | PROCESS.md §4 "S9 rescope" | **blocked on design**: a self-minted cap needs a **transfer channel** to reach a peer — the only cross-domain transfers today are parent-mediated spawn grants (which `child_offer` covers). Child-initiated transfer = the flagged detached-peers question; one candidate is handle-typed args over live calls (extend §3.3 boundary translation to the svc dispatch path) |
| **jacl flagged questions** — does self-mint need to cover *detached* peers; is "Endpoint" still the right name for live offers + service points | PROCESS.md §4 rescope tail | awaiting consumer input |
| **Crash handling / `poll` portability** — interp runs a synchronous child lazily, JIT eagerly, so `poll`-based control flow diverges; `$?` = 128+signal mapping lands with convergence | STAGE1.md "Known caveat" | todo — revisit after concurrent stages (eager children shrink the gap) |
| **`fork`/`clone`** — parked-domain clone (full-copy), fork endpoint with duplicated reply token | PROCESS.md S11, STAGE1.md item 7 | parked (S11) |

## Detached-windows residue (core BUILT 2026-07-23)

| item | owner | status |
|---|---|---|
| Minter re-grant / quota **split** down the grant tree (§3.3-style; today the minter is top-level-only) | PROCESS.md §5 | todo |
| Browser/wasm32 detached pool sizing inside one wasm memory | PROCESS.md O4 | open |
| Multi-window (detached-subtree) freeze — consistent cut across windows | PROCESS.md O6, DURABILITY.md R4 | open (durable domains refuse detached spawns fail-closed meanwhile) |

## §3.6 residue (core loop BUILT; recorded with the as-built blocks)

| item | owner | status |
|---|---|---|
| Coverage-remapped **grouped** live bindings for the slot route (flat/identity op mapping today) | IMPORTS.md §3.6 slice 4 block | todo — when a consumer needs subset coverage |
| **Passive-instance deletion** — retire §3.2 v2 instanced offers once live serving covers their uses | IMPORTS.md §3.6 | needs owner sign-off (jacl designs still moving) |
| **Durable event-parks** — a freeze currently fails closed on any `ParkedOn` fiber (its wake is scheduler state no snapshot carries) | IMPORTS.md §3.6 slice 5a block | todo — durability track |
| **Join-in-fiber parks** — `Join` (and `svc.wait`'s own empty-queue park) stay vCPU-level; child-trap propagation into a parked fiber is the open design question | IMPORTS.md §3.6 slice 5a block | todo |
| **Native fast-backend serving** — bytecode/JIT serve loops instead of the `module_serves` oracle fold | IMPORTS.md §3.6 parity block | open **optimization** — awaits benchmark evidence (jacl workloads) or settled semantics; the fold is the differential baseline |
| `module_serves` residual — a JIT parent that spawns a serving child module but never wires it keeps the JIT (child's `svc.*` refuses probeably) | IMPORTS.md separate-module block | recorded; acceptable (any parent that talks to its child folds) |
| "Entry returned" domain persistence — a domain that serves after its entry returns | IMPORTS.md §3.6 | dissolved into design: a domain persists by looping in `svc.wait`; anything else reintroduces the passive instance. Revisit only with a concrete consumer shape |

## Exec residue (v1 BUILT: host / scripted / domain)

| item | owner | status |
|---|---|---|
| Streaming reads before exit (`read_out` pre-`status`), incremental stdin, signals | EXEC.md "reserved, not promised" | reserved extension |
| `domain_exec` trap→exit-code mapping (v1: a trapping child is a failed `run`, `-EINVAL`) | EXEC.md as-built | reserved refinement |
| Guest-served `"exec"` — a parent serving its child's exec with its own code (the none-the-wiser nested shell) | EXEC.md table row 4 | rides Endpoint self-mint |

## Standing (not scheduled)

| item | owner | status |
|---|---|---|
| I33 flake (jit_killpath_stops_runaway_child, macOS) | ISSUES.md | another agent's |
| §3 substrate generalization (`create(module, window, budget)` subsuming the op-15 surface) | PROCESS.md §3 | future; op-15 chosen deliberately as the incremental form |
| `Budget` charging at create (the passable object exists; per-domain charging is the follow-up) | PROCESS.md §5 / cap_id::BUDGET docs | todo |
