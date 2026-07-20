//! IMPORTS.md §3.2 slice 2 — the **wiring primitive** for provider-side interface offers:
//!
//! * `intern_interface` — the per-`Host` structural interface intern (D59 applied to
//!   capability interfaces): id-equality ≡ structural equality of the op-signature list;
//! * `wire_impl` — the authority-moving act: mint a `Binding::GuestImpl` entry whose op
//!   signatures are *derived* from the offered functions' declared types, fail-closed;
//! * `bound_import_for_impl` — the wiring-time structural signature check that binds an
//!   import slot to one op of a wired offer;
//! * execution (slice 3): a wired op runs through the **generic dispatch** as a v1 pure
//!   dispatch — a fresh reference run over the offer's functions, windowless, empty
//!   powerbox, fixed fuel — so all three backends share one implementation; a wired offer
//!   stays non-durable (out-of-line function-table reference), refused at capture and
//!   drained cleanly.

use std::sync::Arc;
use svm_interp::{iface, Host, NonDurableKind, Trap, Value};
use svm_ir::{BinOp, Block, Func, FuncType, Inst, IntTy, LoadOp, Terminator, ValType};

/// A one-block leaf `(params) -> (results)` whose body just returns its first param (or
/// nothing) — enough to carry a distinct declared signature.
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

/// The offering module's function table for these tests. Func 1 actually computes
/// (`a + b`), so execution tests can observe a real result; func 3 loads from memory, the
/// thing a v1 pure dispatch must fault on (the impl runs windowless).
fn offer_funcs() -> Arc<[Func]> {
    let add = Func {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64],
            insts: vec![Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 0,
                b: 1,
            }],
            term: Terminator::Return(vec![2]),
        }],
    };
    let loads = Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![Inst::Load {
                op: LoadOp::I64,
                addr: 0,
                offset: 0,
                align: 8,
            }],
            term: Terminator::Return(vec![1]),
        }],
    };
    vec![
        leaf(vec![ValType::I64], vec![ValType::I64]), // 0: identity
        add,                                          // 1: a + b
        leaf(vec![], vec![]),                         // 2: unit
        loads,                                        // 3: reads the (absent) window
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
fn a_wired_op_executes_through_the_generic_dispatch() {
    // Slice 3: op dispatch runs the offer's function as a v1 pure dispatch — args in,
    // results out, computed by actual guest code.
    let mut h = Host::new();
    let funcs = offer_funcs();
    let handle = h.wire_impl(&funcs, &[1, 0]).expect("offer");
    let tid = h.resolve_guest_impl(handle).unwrap().type_id;
    // op 0 = add(a, b).
    assert_eq!(
        h.cap_dispatch_slots(tid, 0, handle, &[40, 2], None),
        Ok(vec![42])
    );
    // op 1 = identity.
    assert_eq!(
        h.cap_dispatch_slots(tid, 1, handle, &[7], None),
        Ok(vec![7])
    );
    // Fail-closed legs: op past the list, wrong arity.
    assert!(matches!(
        h.cap_dispatch_slots(tid, 2, handle, &[1], None),
        Err(Trap::CapFault)
    ));
    assert!(matches!(
        h.cap_dispatch_slots(tid, 0, handle, &[1], None),
        Err(Trap::CapFault)
    ));
}

#[test]
fn a_wired_impl_is_windowless_and_powerboxless() {
    // The v1 pure dispatch grants the impl exactly nothing: a load faults (no window) and
    // the impl cannot reach the wiring domain's capabilities (fresh empty powerbox) — the
    // caller's call traps, fail-closed.
    let mut h = Host::new();
    h.grant_clock(); // live caps in the wiring domain, unreachable from the impl
    let funcs = offer_funcs();
    let handle = h.wire_impl(&funcs, &[3]).expect("offer");
    let tid = h.resolve_guest_impl(handle).unwrap().type_id;
    assert!(
        h.cap_dispatch_slots(tid, 0, handle, &[0], None).is_err(),
        "a load inside a windowless impl must trap"
    );
}

#[test]
fn a_wired_import_slot_runs_on_both_engines() {
    // End-to-end: a module imports "adder", the host wires an offer into the slot, and the
    // guest's `call.import` computes through the wired guest impl — identically on the
    // tree-walker and the bytecode engine (the JIT thunk shares the same generic dispatch;
    // its harness lives with the svm-run wiring surface).
    let m = svm_text::parse_module(
        "import 0 \"adder\" (i64, i64) -> (i64)\n\
         func (i64, i64) -> (i64) {\n\
         block0(va: i64, vb: i64):\n\
           vh = i32.const 0\n\
           vr = call.import 0 vh (va, vb)\n\
           return vr\n\
         }\n",
    )
    .expect("parse");
    svm_verify::verify_module(&m).expect("verifies");

    let build_host = || {
        let mut h = Host::new();
        let handle = h.wire_impl(&offer_funcs(), &[1]).expect("offer");
        let b = h
            .bound_import_for_impl(handle, 0, &m.imports[0].sig, false)
            .expect("slot sig matches the offer op");
        h.set_import_bindings(vec![b]);
        h
    };

    let args = [Value::I64(40), Value::I64(2)];
    let mut fuel_a = 1_000_000u64;
    let mut host_a = build_host();
    let tree = svm_interp::run_with_host(&m, 0, &args, &mut fuel_a, &mut host_a);
    assert_eq!(tree, Ok(vec![Value::I64(42)]), "tree-walker");

    let mut fuel_b = 1_000_000u64;
    let mut host_b = build_host();
    let byte =
        svm_interp::bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel_b, &mut host_b)
            .expect("module is bytecode-eligible");
    assert_eq!(byte, Ok(vec![Value::I64(42)]), "bytecode engine");
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
