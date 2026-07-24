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
| **Native fast-backend serving** — bytecode + JIT serve loops; the `module_serves` fold now covers only what still needs the oracle | ISSUES.md I36, IMPORTS.md §3.6 parity block | **serve loops BUILT 2026-07-24** (bytecode slices 1–2: poll/wait + caller parking + `child_offer`; JIT slice 3: poll/wait over handler trampolines) — remaining: JIT caller side (op 14 + persistent child Hosts), bytecode granted spawns 8/11/13; the fold stays as the differential baseline |
| **Serving-cluster sequencing (owner-agreed 2026-07-24):** I36 bytecode loop → I36 JIT loop → I41 graceful revocation (small, can interleave) → multi-consumer `svc.wait` → durable serving; poison-drain (I37 escalation a) slots in wherever the softer failure mode is wanted | this table + ISSUES.md I36–I41 | the build order for everything above — capture serve state once, against the native loops |
| **Endpoint self-mint** — a domain minting a live offer over its *own* impl-export to hand to an arbitrary peer | PROCESS.md §4 "S9 rescope" (decision block) | **DROPPED 2026-07-23 (owner decision)** — discovery/coordination with arbitrary domains is out; every capability transfer stays **mediated by an authority in the grant graph** (spawn grants + `child_offer` regrant, all built). Note the mediation-consistent residue that survives: a domain minting an offer over its *own* export to grant **down its own grant graph** (a parent serving its children) — see the guest-served exec row |
| ~~**jacl flagged questions**~~ | PROCESS.md §4 rescope tail | **answered 2026-07-23 (owner)** — (1) no self-mint / no peer transfer channel: authority-mediated only; (2) "Endpoint" retired as a concept name — the substrate keeps its low-level nouns (live offers, service points, handler returns); the serve/reply-split drop is confirmed |
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
| **Durable event-parks** — a freeze currently fails closed on any `ParkedOn` fiber (its wake is scheduler state no snapshot carries) | IMPORTS.md §3.6 slice 5a block | todo — durability track (part of the serving-review sweep, ISSUES.md I36–I40 lead-in) |
| **Join-in-fiber parks** — `Join` (and `svc.wait`'s own empty-queue park) stay vCPU-level; child-trap propagation into a parked fiber is the open design question | IMPORTS.md §3.6 slice 5a block | todo (ditto) |
| **Serving-review items** — handler-trap blast radius (supervision/sharding idioms to document; poll→detach, never join a trapped child), servicer-side admission fairness, serialized-handler ceiling (document next to F6), `svc_results` orphan sweep | ISSUES.md **I37 / I38 / I39 / I40 / I41** | **I41 BUILT 2026-07-24** (revoked-once-valid handle → probeable `-EBADF` on all three backends, storage-free tombstone; forged still traps); I38 + I40 remain small substrate slices, I37 + I39 documentation/pattern work |
| **Multi-consumer `svc.wait`** — N spawned server vCPUs drain one domain queue (the I39 opt-in ladder's top rung: parallel dispatch for domains that accept the threading discipline) | ISSUES.md I39 resolution path | **oracle BUILT 2026-07-24** (multi-waiter `svc_waiters` + wake-all; **timed `svc.wait`** as the wind-down primitive — the I38 sketch, pulled in as this rung's prerequisite; plus a pre-existing settle/park TOCTOU fixed by the atomic `cap_reply_or_stash`; semantics pinned + hammered in `svc_multi_consumer.rs`). Fast backends still veto svc+threads and the timed form; their rung lands when a consumer demands it |
| **Durable serving domains** — freeze/thaw a domain parked in `svc.wait` with in-flight handler parks; first target: **subtree-freeze-as-unit** (server + callers in one cut, so tickets/queue are internal data), cross-cut callers get the O10 re-issue rule; needs stable domain keys (not `Arc` ptrs) + event-park capture | TODO §3.6 residue (durable event-parks), PROCESS.md O10 | **consumer interest registered 2026-07-24** (pairs with I37: thaw-from-snapshot instead of cold respawn after a trap) — sequence after I36 so serve-state capture is built once against the native loops |
| `module_serves` residual — a JIT parent that spawns a serving child module but never wires it keeps the JIT (child's `svc.*` refuses probeably) | IMPORTS.md separate-module block | recorded; acceptable (any parent that talks to its child folds) |
| "Entry returned" domain persistence — a domain that serves after its entry returns | IMPORTS.md §3.6 | dissolved into design: a domain persists by looping in `svc.wait`; anything else reintroduces the passive instance. Revisit only with a concrete consumer shape |

## Exec residue (v1 BUILT: host / scripted / domain)

| item | owner | status |
|---|---|---|
| Streaming reads before exit (`read_out` pre-`status`), incremental stdin, signals | EXEC.md "reserved, not promised" | reserved extension |
| `domain_exec` trap→exit-code mapping (v1: a trapping child is a failed `run`, `-EINVAL`) | EXEC.md as-built | reserved refinement |
| Guest-served `"exec"` — a parent serving its child's exec with its own code (the none-the-wiser nested shell) | EXEC.md table row 4 | re-scoped 2026-07-23: self-mint-as-transfer is dropped, but this needs only the **mediation-consistent** form — a parent minting an offer over its *own* export and spawn-granting it to its child (granter has authority over grantee; no new transfer channel) |

## Standing (not scheduled)

| item | owner | status |
|---|---|---|
| I33 flake (jit_killpath_stops_runaway_child, macOS) | ISSUES.md | another agent's |
| §3 substrate generalization (`create(module, window, budget)` subsuming the op-15 surface) | PROCESS.md §3 | future; op-15 chosen deliberately as the incremental form |
| `Budget` charging at create (the passable object exists; per-domain charging is the follow-up) | PROCESS.md §5 / cap_id::BUDGET docs | todo |
