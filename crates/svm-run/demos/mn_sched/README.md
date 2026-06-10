# mn_sched — a guest-built M:N green-thread scheduler

```
cargo run -p svm-run -- crates/svm-run/demos/mn_sched/mn_sched.c   # prints 1024
```

Proof that the VM's concurrency **primitives compose into a real M:N runtime with no scheduler
baked into the VM** (DESIGN D56/D57; see `SCHEDULING.md`). The VM ships only vCPUs
(`thread.spawn`, 1:1 OS threads), stackful fibers (`cont.*`), and the futex + atomics — **this
program is the scheduler**.

Shape: 4 worker OS-threads, each cooperatively round-robining 8 green tasks (fibers) that yield
and increment one shared atomic — `4 × 8 × 32 = 1024`. Tasks are pinned to their worker (fibers
are thread-affine by design, D57), so this is *sharded / thread-per-core* M:N (glommio/seastar
style). The total is interleaving-invariant, so it's identical on the interpreter (the M:N
deterministic oracle) and the JIT (real OS threads) — pinned by
`c_frontend::c_guest_mn_scheduler_demo`.

Roadmap (`SCHEDULING.md`): a stackless **work-stealing** variant (no VM change), then a
*migratable-fiber* primitive for stackful work-stealing (D57, gated on a loom-verified ownership
protocol).
