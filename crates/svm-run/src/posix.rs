//! The **POSIX personality as a powerbox `HostCap`** — the bridge that lets the LLVM on-ramp world
//! (`svm-llvm` guests, which reach the host through `__vm_cap_resolve` + `__vm_host_call`) use the
//! `svm-posix` process/fd/signal ABI (POSIX.md), the same surface the chibicc/name-binding world already
//! links against.
//!
//! `svm-posix` normally grants itself on a `Host` by name binding ([`svm_posix::grant`], the chibicc
//! path). The on-ramp instead resolves host capabilities by **name** and calls them by op number, exactly
//! as `demos/postgres/os_shim.c` reaches the `fs` cap. [`posix_cap`] wraps the personality's [`HostProc`]
//! factory ([`svm_posix::cap`]) in a [`HostCap`] at [`svm_interp::cap_id::HOST_PROC`], so an embedder can
//! grant it under a name (e.g. `"posix"`) via `Instance::run_with_caps`, and a guest calls
//! `__vm_host_call(__vm_cap_resolve("posix"), OP_*, a, b, c, d)`.

use crate::HostCap;
use svm_posix::Posix;

/// Build the POSIX personality as a named powerbox capability. Returns the [`HostCap`] to grant (e.g.
/// `("posix", cap)` in `run_with_caps`) and the shared [`Posix`] handle the embedder keeps to read the
/// captured output, wire the spawn delegate ([`Posix::set_spawn`]), or raise a signal
/// ([`Posix::raise_signal`]). `heap_base`/`heap_end` bound the window-heap region the personality's
/// `malloc` hands out (both window offsets, within the guest window and clear of its data/stack); pass
/// `0, 0` when the guest brings its own allocator and only uses the process/fd/signal ops. `stdin`
/// preloads standard input for `read(0, …)`.
///
/// The cap grants once per backend over one shared personality state, so an interp/JIT differential run
/// observes the same `Posix` — read it back after the run.
pub fn posix_cap(heap_base: u64, heap_end: u64, stdin: Vec<u8>) -> (HostCap, Posix) {
    posix_cap_inner(heap_base, heap_end, stdin, false)
}

/// [`posix_cap`] with the **#797 controlling terminal** enabled at grant time
/// ([`Posix::enable_terminal`]): fds 0-2 answer `isatty`, `read(0)` rides the terminal input pipe
/// (parks empty — a real prompt), and the embedder types with [`Posix::feed_terminal`] from
/// another thread while the run is live (the `run_interp_terminal` witness shape). This is the
/// lane an **interactive** on-ramp guest (bash `-i`) runs on.
pub fn posix_cap_terminal(heap_base: u64, heap_end: u64) -> (HostCap, Posix) {
    posix_cap_inner(heap_base, heap_end, Vec::new(), true)
}

fn posix_cap_inner(
    heap_base: u64,
    heap_end: u64,
    stdin: Vec<u8>,
    terminal: bool,
) -> (HostCap, Posix) {
    let (posix, make) = svm_posix::cap(heap_base, heap_end, stdin);
    // #863 — the personality's own fork factory rides along, so a `fork()` through the powerbox
    // path gets real POSIX semantics (own fd table/cwd/env/signals, shared memfs), not a shared blob.
    let fork = svm_posix::cap_fork_factory(&posix);
    // #802 slice 4 — the grant installs everything `svm_posix::grant` installs on the name-binding
    // path, not just the handler: the async-signal door (which also carries the #799
    // caller-request door — without it `fork` (op 51) has no park request and returns `-ENOSYS`),
    // the #972 exec-remap hook, and the #801 op vtable (so an exec'd image's `__px_*` manifest
    // binds through the coverage walk).
    let p = posix.clone();
    let cap = HostCap::custom(svm_interp::cap_id::HOST_PROC, 0, move |h, _win| {
        let handle = h.grant_host_proc_forkable(make(), std::sync::Arc::clone(&fork));
        let (door, armed) = svm_posix::cap_signal_source(&p);
        h.set_signal_source(door, armed);
        h.push_exec_remap_hook(svm_posix::cap_exec_remap_hook(&p));
        let (names, sigs) = svm_posix::cap_vtable();
        h.set_host_proc_vtable(handle, names, sigs);
        if terminal {
            p.enable_terminal(h);
        }
        handle
    });
    (cap, posix)
}

/// The **`net` capability** over an existing personality (POSIX.md §5a) — grant it alongside the
/// posix cap under its own name (e.g. `("net", net_cap(&posix))` in `run_with_caps`). Socket fds it
/// mints live in the same fd table, so the guest reads/writes them through the posix cap's ordinary
/// `read`/`write` ops. Loopback is served in-personality (the memnet); anything beyond routes to the
/// embedder's [`svm_posix::NetDelegate`] ([`Posix::set_net`]) or fails closed.
pub fn net_cap(posix: &Posix) -> HostCap {
    HostCap::host_proc(0, svm_posix::net_cap_factory(posix))
}
