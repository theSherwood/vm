//! DURABILITY.md §12.8 Phase 4 Slice A.7 — **parked-vCPU / `Blocking.work` latency**. A durable
//! stop-the-world freeze waits for every vCPU to reach a safepoint; a vCPU inside a host `Blocking`
//! call has no poll site, so the freeze would otherwise stall for the whole (latency-unbounded) call
//! (R6). The fail-closed cut: once an async freeze has landed (the global freeze word reads
//! `UNWINDING`), a durable vCPU **refuses to enter** a new blocking host call (`Trap::ThreadFault`),
//! so snapshot latency excludes new host calls once a freeze is requested. Cancelling an *already
//! in-flight* call is deferred (R2).
//!
//! The gate lives in the **shared** capability dispatch (`Host::cap_dispatch_slots`) that *both*
//! backends funnel a `cap.call` through (the JIT via `svm-run`'s `cap_thunk`), so exercising it on the
//! interpreter covers the JIT's blocking path too — and deterministically, without racing an async
//! controller against a real OS thread.

use std::time::Duration;
use svm_interp::{iface, GuestMem, Host, Trap, STATE_NORMAL, STATE_OFF, STATE_UNWINDING};

/// A trivial flat window: the gate only reads the 4-byte freeze word at `STATE_OFF`, and a `Blocking`
/// op is window-independent, so a `Vec` backing is all the dispatch needs.
struct VecMem(Vec<u8>);

impl GuestMem for VecMem {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        let (p, l) = (ptr as usize, len as usize);
        self.0.get(p..p + l).map(<[u8]>::to_vec)
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        let p = ptr as usize;
        self.0.get_mut(p..p + data.len())?.copy_from_slice(data);
        Some(())
    }
}

fn set_state(mem: &mut VecMem, state: i32) {
    mem.write_bytes(STATE_OFF, &state.to_le_bytes())
        .expect("state word fits the window");
}

/// A direct `Blocking.work` cap.call fails **closed** the moment an async freeze has landed, and runs
/// normally once the window is back to `NORMAL` — the §12.8 4A.7 fail-closed, deterministically.
#[test]
fn blocking_call_fails_closed_once_a_freeze_has_landed() {
    let mut host = Host::new();
    host.set_durable(true);
    let h = host.grant_blocking(Duration::ZERO, None);
    let mut mem = VecMem(vec![0u8; 4096]);

    // Freeze landed (UNWINDING): entering a new blocking offload would stall the STW, so refuse.
    set_state(&mut mem, STATE_UNWINDING);
    let refused = host.cap_dispatch_slots(iface::BLOCKING, 0, h, &[7], Some(&mut mem));
    assert!(
        matches!(refused, Err(Trap::ThreadFault)),
        "freeze landed → blocking call must fail closed, got {refused:?}",
    );

    // No freeze (NORMAL): the same call runs and returns its one deterministic result.
    set_state(&mut mem, STATE_NORMAL);
    let ran = host.cap_dispatch_slots(iface::BLOCKING, 0, h, &[7], Some(&mut mem));
    assert!(
        matches!(&ran, Ok(v) if v.len() == 1),
        "no freeze → blocking call must run, got {ran:?}",
    );
}

/// The gate is conditioned on a **durable** domain: a non-durable run's byte at window offset 0 is
/// ordinary guest data, not a freeze word, and must never spuriously refuse a blocking call.
#[test]
fn blocking_call_runs_when_not_durable_even_if_offset_zero_looks_like_unwinding() {
    let mut host = Host::new(); // NOT durable
    let h = host.grant_blocking(Duration::ZERO, None);
    let mut mem = VecMem(vec![0u8; 4096]);
    set_state(&mut mem, STATE_UNWINDING); // a coincidental guest byte pattern, not a freeze word

    let ran = host.cap_dispatch_slots(iface::BLOCKING, 0, h, &[7], Some(&mut mem));
    assert!(
        matches!(&ran, Ok(v) if v.len() == 1),
        "non-durable: offset-0 byte is guest data, the blocking call must run, got {ran:?}",
    );
}
