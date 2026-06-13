# steal_fibers — work-stealing over stackful, migratable fibers (Demo 3)

```
cargo run -p svm-run -- crates/svm-run/demos/steal_fibers/steal_fibers.c   # prints 256, 121920
```

The capstone of the migratable-fiber track (D57; design + verification story in DESIGN.md §23):
the work-stealing scheduler from `demos/work_stealing` with its state-machine structs replaced by
**fibers**. Task handles sit in a guest injector queue + per-worker deques; an idle worker steals a
*suspended fiber* from a busy sibling and resumes it on its own OS thread — on the JIT a genuine
cross-thread native-stack switch (claimed through the loom-verified `Ownership` word), on the
interpreter a pure-data `Vec<Frame>` hand-off. Go-class scheduling of arbitrary, unmodified code,
entirely as guest policy: the VM contributed only the domain-wide handle namespace and the
single-owner resume arbiter.

What makes it distinctly *stackful*: the task yields from **inside a nested call frame**
(`step_in_callee`) — inexpressible for a stackless state machine (function coloring) — and computes
its return value from locals held live across every yield and migration. So the second printed
total (`121920`, the sum of returns) is a stack-integrity check across all the migrations, not just
a work count (`256`). Both totals are interleaving-invariant ⇒ identical on the interpreter (the
deterministic M:N oracle) and the JIT (real OS threads). Pinned by
`c_frontend::c_guest_steal_fibers_demo` + `run::demo_steal_fibers_runs`.
