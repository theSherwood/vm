# mn_sched — a guest-built M:N green-thread scheduler

```
cargo run -p svm-run -- crates/svm-run/demos/mn_sched/mn_sched.c   # prints 1024
```

Proof that the VM's concurrency **primitives compose into a real M:N runtime with no scheduler
baked into the VM** (DESIGN D56/D57; see DESIGN.md §23). The VM ships only vCPUs
(`thread.spawn`, 1:1 OS threads), stackful fibers (`cont.*`), and the futex + atomics — **this
program is the scheduler**.

Shape: 4 worker OS-threads, each cooperatively round-robining 8 green tasks (fibers) that yield
and increment one shared atomic — `4 × 8 × 32 = 1024`. Tasks are pinned to their worker **by this scheduler's choice** (fibers are
migratable since D57 — affinity is guest policy here), so this is *sharded / thread-per-core*
M:N (glommio/seastar style). The total is interleaving-invariant, so it's identical on the interpreter (the M:N
deterministic oracle) and the JIT (real OS threads) — pinned by
`c_frontend::c_guest_mn_scheduler_demo`.

The track this opened is complete (DESIGN.md §23): `demos/work_stealing` is the stackless
work-stealing variant (no VM change), and `demos/steal_fibers` is **stackful work-stealing over
migratable fibers** (D57, the loom-verified ownership protocol + empirical net).
