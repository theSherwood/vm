# work_stealing — a guest-built work-stealing M:N scheduler (stackless tasks)

```
cargo run -p svm-run -- crates/svm-run/demos/work_stealing/work_stealing.c   # prints 256
```

Demo 2 of the concurrency-validation track (DESIGN D56/D57; see DESIGN.md §23). The companion to
`demos/mn_sched` (which is *sharded, stackful* — fibers pinned per worker). Here a task is a
**state machine** (a plain struct; its resume state is a field), so it is just **data** and moves
freely between worker threads. That makes **work-stealing** possible **with no VM change**: an idle
worker steals a task from a busy sibling (or pulls from a global injector) and resumes it on its own
thread — a pointer hand-off, safe by construction (no native stack to migrate).

Architecture (tokio-style): a global injector queue + per-worker deques + stealing. Built entirely
from the VM's primitives — vCPUs (`thread.spawn`), the futex (under `pthread_mutex`), and atomics;
**no fibers, no scheduler in the VM**.

The grand total (`NTASKS · STEPS = 16 · 16 = 256`) is interleaving-invariant — and proves no task
was lost or double-run as they migrated — so it is identical on the interpreter (the M:N
deterministic oracle) and the JIT (real OS threads), regardless of *which* worker ran each task.
Pinned by `c_frontend::c_guest_work_stealing_demo`.

The migratable-fiber primitive (D57) has since brought this architecture to *stackful* tasks too —
see `demos/steal_fibers` (Demo 3). Stackless remains the natural substrate for the async I/O
ring (B) and anything that must move tasks without native stacks.
