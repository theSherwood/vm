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

use svm_interp::{DurableBinding, DurableHandle, Host, NonDurableKind, StreamRole};

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
