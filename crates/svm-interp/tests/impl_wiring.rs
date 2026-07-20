//! IMPORTS.md §3.2 slice 2 — the **wiring primitive** for provider-side interface offers:
//!
//! * `intern_interface` — the per-`Host` structural interface intern (D59 applied to
//!   capability interfaces): id-equality ≡ structural equality of the op-signature list;
//! * `wire_impl` — the authority-moving act: mint a `Binding::GuestImpl` entry whose op
//!   signatures are *derived* from the offered functions' declared types, fail-closed;
//! * `bound_import_for_impl` — the wiring-time structural signature check that binds an
//!   import slot to one op of a wired offer;
//! * inertness + durability: the generic dispatch `CapFault`s on a wired offer (guest code
//!   is executor-serviced — the slice-3 eval-loop routing), and a wired offer is
//!   non-durable (out-of-line domain reference), refused at capture and drained cleanly.
//!
//! Execution of a wired op (the trampoline into the offering domain) is slice 3; nothing
//! here runs guest code.

use std::sync::Arc;
use svm_interp::{iface, Host, NonDurableKind, Trap};
use svm_ir::{Block, Func, FuncType, Terminator, ValType};

/// A one-block leaf `(params) -> (results)` whose body just returns its first param (or
/// nothing) — enough to carry a distinct declared signature; nothing here executes it.
fn leaf(params: Vec<ValType>, results: Vec<ValType>) -> Func {
    let term = if results.is_empty() {
        Terminator::Return(vec![])
    } else {
        Terminator::Return(vec![0])
    };
    Func {
        params: params.clone(),
        results,
        blocks: vec![Block {
            params,
            insts: vec![],
            term,
        }],
    }
}

fn sig(params: Vec<ValType>, results: Vec<ValType>) -> FuncType {
    FuncType { params, results }
}

/// The offering module's function table for these tests: two distinct op shapes.
fn offer_funcs() -> Arc<[Func]> {
    vec![
        leaf(vec![ValType::I64], vec![ValType::I64]), // 0
        leaf(vec![ValType::I64, ValType::I64], vec![ValType::I64]), // 1
        leaf(vec![], vec![]),                         // 2
    ]
    .into()
}

#[test]
fn intern_is_structural_and_allocates_from_the_base() {
    let mut h = Host::new();
    let a = vec![sig(vec![ValType::I64], vec![ValType::I64])];
    let b = vec![sig(vec![ValType::I64], vec![ValType::I64])];
    let c = vec![sig(vec![ValType::I32], vec![ValType::I64])];
    let ia = h.intern_interface(&a);
    assert!(
        ia >= iface::GUEST_IMPL_BASE,
        "guest ids allocate above the built-ins"
    );
    assert_eq!(
        ia,
        h.intern_interface(&b),
        "structurally identical lists collide to the same id (D59)"
    );
    assert_ne!(
        ia,
        h.intern_interface(&c),
        "structurally distinct lists get distinct ids"
    );
    // Interning is stable: re-asking never re-allocates.
    assert_eq!(ia, h.intern_interface(&a));
}

#[test]
fn wire_impl_derives_sigs_and_mints_a_resolvable_handle() {
    let mut h = Host::new();
    let funcs = offer_funcs();
    let handle = h.wire_impl(&funcs, &[1, 0]).expect("well-formed offer");
    let entry = h.resolve_guest_impl(handle).expect("handle resolves");
    // Op order is the offer's, and each op's signature IS the named function's declared type.
    assert_eq!(&*entry.ops, &[1, 0]);
    assert_eq!(
        &*entry.sigs,
        &[
            sig(vec![ValType::I64, ValType::I64], vec![ValType::I64]),
            sig(vec![ValType::I64], vec![ValType::I64]),
        ]
    );
    assert!(entry.type_id >= iface::GUEST_IMPL_BASE);

    // Two offers with the same shape share a type_id (structural identity); a different
    // shape gets a fresh one.
    let same = h.wire_impl(&funcs, &[1, 0]).expect("second offer");
    let other = h.wire_impl(&funcs, &[2]).expect("distinct offer");
    let tid = h.resolve_guest_impl(handle).unwrap().type_id;
    assert_eq!(h.resolve_guest_impl(same).unwrap().type_id, tid);
    assert_ne!(h.resolve_guest_impl(other).unwrap().type_id, tid);
}

#[test]
fn wire_impl_fails_closed() {
    let mut h = Host::new();
    let funcs = offer_funcs();
    assert!(h.wire_impl(&funcs, &[]).is_none(), "empty op list");
    assert!(h.wire_impl(&funcs, &[0, 9]).is_none(), "op out of range");
    // Nothing was minted by the refusals: a fresh wire still works and a forged handle
    // still resolves nowhere.
    assert!(h.wire_impl(&funcs, &[0]).is_some());
    assert!(matches!(h.resolve_guest_impl(0x7f), Err(Trap::CapFault)));
}

#[test]
fn bound_import_for_impl_checks_the_slot_signature_structurally() {
    let mut h = Host::new();
    let funcs = offer_funcs();
    let handle = h.wire_impl(&funcs, &[1, 0]).expect("offer");
    let declared = sig(vec![ValType::I64], vec![ValType::I64]); // matches op 1 (funcs[0])

    let b = h
        .bound_import_for_impl(handle, 1, &declared, false)
        .expect("matching declaration binds");
    assert_eq!(b.op, 1);
    assert_eq!(b.handle, handle);
    assert!(b.bound && !b.rebindable);
    assert_eq!(b.type_id, h.resolve_guest_impl(handle).unwrap().type_id);

    let r = h
        .bound_import_for_impl(
            handle,
            0,
            &sig(vec![ValType::I64, ValType::I64], vec![ValType::I64]),
            true,
        )
        .expect("rebindable binds too");
    assert!(r.bound && r.rebindable);

    // Fail-closed legs: sig mismatch, op past the list, forged handle.
    assert!(h
        .bound_import_for_impl(handle, 0, &declared, false)
        .is_none());
    assert!(h
        .bound_import_for_impl(handle, 2, &declared, false)
        .is_none());
    assert!(h.bound_import_for_impl(0x7f, 1, &declared, false).is_none());
}

#[test]
fn generic_dispatch_on_a_wired_offer_is_inert() {
    // Guest code is executor-serviced (the slice-3 eval-loop routing); the generic dispatch
    // must treat a wired offer as an inert `CapFault`, exactly like the other
    // executor-dispatch interfaces.
    let mut h = Host::new();
    let funcs = offer_funcs();
    let handle = h.wire_impl(&funcs, &[0]).expect("offer");
    let tid = h.resolve_guest_impl(handle).unwrap().type_id;
    assert!(matches!(
        h.cap_dispatch_slots(tid, 0, handle, &[1], None),
        Err(Trap::CapFault)
    ));
}

#[test]
fn a_wired_offer_is_non_durable_and_drains_cleanly() {
    let mut h = Host::new();
    let funcs = offer_funcs();
    let handle = h.wire_impl(&funcs, &[0]).expect("offer");

    // Freeze refuses while the offer is live (all-or-nothing), naming the kind.
    let refusal = h.capture_durable_handles().expect_err("non-durable");
    assert_eq!(refusal.kind, NonDurableKind::GuestImpl);

    // Draining closes the slot; the guest-held handle value is then inert (D37: the
    // generation is retained, never recycled into a false positive).
    let drained = h.drain_non_durable();
    assert!(drained.iter().any(|d| d.kind == NonDurableKind::GuestImpl));
    assert!(matches!(h.resolve_guest_impl(handle), Err(Trap::CapFault)));
    h.capture_durable_handles()
        .expect("table is snapshottable once drained");
}
