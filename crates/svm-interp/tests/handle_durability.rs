//! Handle-table durability classification + `(slot, generation)` pinning (DURABILITY.md
//! §12.5). These exercise the `Host` primitives the snapshot codec builds on:
//!
//! * `capture_durable_handles` — the live table classified into the re-grantable §12.5 set,
//!   or a clean refusal if any live slot is non-durable (freeze is all-or-nothing);
//! * `restore_durable_handles` / `grant_at` — reinstating that set into a fresh table while
//!   pinning each `(slot, generation)`, so a guest-held handle value (`(generation << 8) |
//!   slot`) still resolves after restore.
//!
//! The invariant tested is structural: a handle value is a pure function of `(generation,
//! slot)` and resolve re-checks `type_id` + `generation`, so equality of the captured set
//! across a restore ⇒ every guest-held handle stays valid. (The full freeze→serialize→
//! restore→thaw run lands with the snapshot-codec slice that wires this to the window image.)

use svm_interp::{cap_id, DurableBinding, DurableHandle, Host, NonDurableKind, StreamRole, Trap};

/// Grant a spread of durable bindings, capture, restore into a fresh table, and confirm the
/// captured set is byte-for-byte identical — slot, generation, type_id, and binding all pinned.
#[test]
fn durable_handles_round_trip_through_capture_restore() {
    let mut a = Host::new();
    a.grant_clock();
    a.grant_stream(StreamRole::Out);
    a.grant_memory();
    a.grant_exit();
    a.grant_address_space(0x2000, 0x1000);
    a.grant_instantiator(0x0, 0x4000);

    let captured = a
        .capture_durable_handles()
        .expect("every binding is durable");
    assert_eq!(captured.len(), 6, "all six live slots captured");
    // Ascending slot order, contiguous from 0 (grants fill the first free slots).
    assert_eq!(captured[0].slot, 0);
    assert!(
        captured.windows(2).all(|w| w[0].slot < w[1].slot),
        "ascending slot order"
    );
    // Value-typed bindings survive verbatim.
    assert_eq!(captured[0].binding, DurableBinding::Clock);
    assert_eq!(captured[1].binding, DurableBinding::Stream(StreamRole::Out));
    assert!(captured.iter().any(|h| h.binding
        == DurableBinding::AddressSpace {
            base: 0x2000,
            size: 0x1000
        }));

    let mut b = Host::new();
    b.restore_durable_handles(&captured);
    assert_eq!(
        b.capture_durable_handles().unwrap(),
        captured,
        "restore reinstates the exact (slot, generation, type_id, binding) set"
    );
}

/// A fresh table starts every slot at generation 0; restore must pin the *captured*
/// generation, not whatever the destination table happens to hold. Bump slot 0's generation
/// via close+re-grant so the distinction is observable.
#[test]
fn restore_pins_generation_not_destination_default() {
    let mut a = Host::new();
    let h0 = a.grant_clock(); // slot 0, generation 1
    a.close(h0);
    a.grant_clock(); // slot 0 again, generation 2 (close kept the generation)

    let captured = a.capture_durable_handles().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].slot, 0);
    assert_eq!(
        captured[0].generation, 2,
        "re-grant after close advanced the generation"
    );

    let mut b = Host::new(); // slot 0 is generation 0 here
    b.restore_durable_handles(&captured);
    let restored = b.capture_durable_handles().unwrap();
    assert_eq!(
        restored, captured,
        "generation 2 pinned, not reset to the fresh table's 0"
    );
}

/// A live non-durable handle (here an `IoRing`, which carries out-of-line host state) makes
/// the table non-snapshottable: capture refuses, naming the offending slot, rather than
/// dropping the authority.
#[test]
fn capture_refuses_a_non_durable_handle() {
    let mut a = Host::new();
    a.grant_clock(); // slot 0, durable
    a.grant_io_ring(); // slot 1, NOT durable (Binding::IoRing carries a ring index)

    let err = a
        .capture_durable_handles()
        .expect_err("an io_ring handle blocks the snapshot");
    assert_eq!(err.slot, 1);
    assert_eq!(err.kind, NonDurableKind::IoRing);
}

/// Draining the non-durable handles turns a capture refusal into a successful one: the out-of-line
/// bindings (an `IoRing` and a `HostFn`) are closed, the durable `Clock` is kept, and
/// `capture_durable_handles` then succeeds. The drained set comes back in ascending slot order so the
/// embedder can audit the relinquished authority (DURABILITY.md §12.5 handle hardening).
#[test]
fn drain_non_durable_makes_a_domain_snapshottable() {
    let mut a = Host::new();
    a.grant_clock(); // slot 0 — durable
    a.grant_io_ring(); // slot 1 — non-durable
    a.grant_host_fn(Box::new(|_op, _args, _mem| Ok(vec![0]))); // slot 2 — non-durable

    assert!(
        a.capture_durable_handles().is_err(),
        "a live non-durable handle blocks the snapshot"
    );

    let drained = a.drain_non_durable();
    assert_eq!(drained.len(), 2, "the io_ring and the host_fn were drained");
    assert_eq!(
        (drained[0].slot, drained[0].kind),
        (1, NonDurableKind::IoRing)
    );
    assert_eq!(
        (drained[1].slot, drained[1].kind),
        (2, NonDurableKind::HostFn)
    );

    let captured = a
        .capture_durable_handles()
        .expect("a drained domain is snapshottable");
    assert_eq!(
        captured.len(),
        1,
        "only the durable Clock survives the drain"
    );
    assert_eq!(captured[0].slot, 0);
    assert_eq!(captured[0].binding, DurableBinding::Clock);
}

/// A drained handle's value is a dead generation: a `cap.call` on it is an inert `CapFault`, never
/// authority into the freed slot (D37). The durable handles the drain left alone still resolve.
#[test]
fn drain_non_durable_kills_stale_handle_values() {
    let mut a = Host::new();
    a.grant_clock(); // durable — kept
    let ring = a.grant_io_ring(); // non-durable — drained

    let drained = a.drain_non_durable();
    assert_eq!(drained.len(), 1, "only the io_ring drained");

    // The drained handle now faults at the use site (freed slot ⇒ resolve fails before the op runs).
    let r = a.cap_dispatch_slots(cap_id::IO_RING, 0, ring, &[], None);
    assert!(
        matches!(r, Err(Trap::CapFault)),
        "a drained handle is a dead generation, got {r:?}"
    );
    // The durable Clock is untouched.
    assert!(a
        .capture_durable_handles()
        .unwrap()
        .iter()
        .any(|h| h.binding == DurableBinding::Clock));
}

/// On an all-durable table draining is a no-op: nothing closes and the captured set is unchanged.
#[test]
fn drain_non_durable_is_a_noop_when_all_durable() {
    let mut a = Host::new();
    a.grant_clock();
    a.grant_memory();
    let before = a.capture_durable_handles().unwrap();

    assert!(
        a.drain_non_durable().is_empty(),
        "nothing non-durable to drain"
    );
    assert_eq!(
        a.capture_durable_handles().unwrap(),
        before,
        "the durable table is unchanged by a no-op drain"
    );
}

/// An empty table captures to an empty set; capacity is the table size, so the codec can
/// bounds-check a captured slot before restore.
#[test]
fn empty_table_captures_empty_and_capacity_is_table_size() {
    let a = Host::new();
    assert_eq!(
        a.capture_durable_handles().unwrap(),
        Vec::<DurableHandle>::new()
    );
    assert_eq!(Host::handle_capacity(), 256);
}
