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
            .bound_import_for_impl(
                handle,
                0,
                m.import_op_sig(0, 0).expect("flat import"),
                false,
            )
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
fn an_offer_regrants_into_a_child_one_hop_deeper() {
    // §3.3 wrap/override: a parent hands a wired offer to a §14 child by name; the child's
    // adopted entry re-interns the (unchanged) structural id and sits one provenance hop
    // deeper. The offer stays executable from the child's own table.
    let mut parent = Host::new();
    let funcs = offer_funcs();
    let handle = parent.wire_impl(&funcs, &[1]).expect("offer");
    let (mut child, _cinst, _cas) = parent
        .spawn_named_child(&[("adder".into(), handle)], 1 << 16)
        .expect("offer handles are re-grantable");
    let ch = child.resolve_cap_name("adder").expect("named in the child");
    let entry = child.resolve_guest_impl(ch).expect("adopted entry");
    assert_eq!(entry.depth, 2, "one re-grant hop past the wiring domain");
    let tid = entry.type_id;
    assert_eq!(
        child.cap_dispatch_slots(tid, 0, ch, &[40, 2], None),
        Ok(vec![42]),
        "the adopted offer executes from the child's table"
    );
}

#[test]
fn child_manifest_binds_named_offers_and_withholds_fail_closed() {
    use svm_ir::ImportMode;
    let mut parent = Host::new();
    let funcs = offer_funcs();
    let handle = parent.wire_impl(&funcs, &[1, 0]).expect("offer");
    let spawn = |parent: &mut Host| {
        parent
            .spawn_named_child(&[("add".into(), handle)], 1 << 16)
            .expect("spawn")
            .0
    };
    // §3.5: manifest signatures live in the type section; build (imports, types) together.
    let manifest = |specs: &[(&str, Vec<ValType>, ImportMode)]| {
        let mut m = svm_ir::Module::default();
        for (name, params, mode) in specs {
            m.add_func_import(*name, sig(params.clone(), vec![ValType::I64]), *mode);
        }
        (m.imports, m.types)
    };

    // A named offer binds the slot to its first signature-matching op (op 0 here: (i64,i64)).
    let mut child = spawn(&mut parent);
    let (imps, tys) = manifest(&[(
        "add",
        vec![ValType::I64, ValType::I64],
        ImportMode::Required,
    )]);
    child
        .bind_child_manifest(&imps, &tys)
        .expect("named offer binds");
    let b = child.import_binding(0).expect("slot bound");
    assert_eq!(b.op, 0, "first sig-matching op");

    // §3.3 withhold: a required import with nothing to bind fails the spawn closed...
    let mut child = spawn(&mut parent);
    let (imps, tys) = manifest(&[("fs", vec![ValType::I64], ImportMode::Required)]);
    assert_eq!(
        child.bind_child_manifest(&imps, &tys),
        Err(0),
        "required + unmatched refuses the manifest"
    );
    // ...a name-matched offer with NO signature-matching op also refuses (never silently binds)...
    let mut child = spawn(&mut parent);
    let (imps, tys) = manifest(&[(
        "add",
        vec![ValType::I32, ValType::I32],
        ImportMode::Required,
    )]);
    assert_eq!(
        child.bind_child_manifest(&imps, &tys),
        Err(0),
        "sig mismatch on a named offer refuses"
    );
    // ...while a rebindable slot just starts empty.
    let mut child = spawn(&mut parent);
    let (imps, tys) = manifest(&[("fs", vec![ValType::I64], ImportMode::Rebindable)]);
    child
        .bind_child_manifest(&imps, &tys)
        .expect("rebindable withhold is an empty slot, not a refusal");
    assert!(child.import_binding(0).is_none(), "slot starts empty");
}

#[test]
fn provenance_reports_platform_vs_ancestor_terminated() {
    // §3.1: `cap.self.provenance(handle)` (self-namespace op 5) — 0 for a platform-native
    // binding, depth d for a wired guest impl, +1 per re-grant hop; forged handles are inert.
    let mut parent = Host::new();
    let clock = parent.grant_clock();
    let funcs = offer_funcs();
    let offer = parent.wire_impl(&funcs, &[1]).expect("offer");

    let prov = |h: &mut Host, cap: i32| {
        h.cap_dispatch_slots(svm_ir::CAP_SELF_TYPE_ID, 5, 0, &[cap as i64], None)
    };
    assert_eq!(prov(&mut parent, clock), Ok(vec![0]), "platform-terminated");
    assert_eq!(
        prov(&mut parent, offer),
        Ok(vec![1]),
        "ancestor-terminated at the wiring domain"
    );

    let (mut child, _, _) = parent
        .spawn_named_child(&[("adder".into(), offer)], 1 << 16)
        .expect("spawn");
    let ch = child.resolve_cap_name("adder").expect("named");
    assert_eq!(
        prov(&mut child, ch),
        Ok(vec![2]),
        "one hop deeper in the child"
    );
    assert!(prov(&mut child, 0x7f).is_err(), "forged handle is inert");
}

/// A stateful provider module: op func 0 bumps a counter in the provider's OWN window and
/// returns the new count — the §3.2 v2 exporter-domain-state probe.
fn counter_provider() -> svm_ir::Module {
    svm_text::parse_module(
        "memory 16\n\
         func () -> (i64) {\n\
         block0():\n\
           va = i64.const 0\n\
           vc = i64.load va\n\
           v1 = i64.const 1\n\
           vn = i64.add vc v1\n\
           i64.store va vn\n\
           return vn\n\
         }\n",
    )
    .expect("provider parses")
}

#[test]
fn an_instanced_offer_keeps_exporter_domain_state_across_calls() {
    let provider = counter_provider();
    svm_verify::verify_module(&provider).expect("provider verifies");
    let mut h = Host::new();
    let offer = h
        .wire_impl_instance(&provider, &[0])
        .expect("instanced offer");
    let tid = h.resolve_guest_impl(offer).unwrap().type_id;
    // The counter lives in the provider's window, not the caller's — successive calls see it.
    for want in 1..=3i64 {
        assert_eq!(
            h.cap_dispatch_slots(tid, 0, offer, &[], None),
            Ok(vec![want]),
            "provider state persists across dispatches"
        );
    }
}

#[test]
fn a_regranted_instanced_offer_shares_one_service_instance() {
    // §3.3 over v2: handing an instanced offer to a child aliases the SAME provider state
    // (like a pipe's shared backing) — parent and child observe one counter.
    let provider = counter_provider();
    svm_verify::verify_module(&provider).expect("verifies");
    let mut parent = Host::new();
    let offer = parent.wire_impl_instance(&provider, &[0]).expect("offer");
    let ptid = parent.resolve_guest_impl(offer).unwrap().type_id;
    assert_eq!(
        parent.cap_dispatch_slots(ptid, 0, offer, &[], None),
        Ok(vec![1])
    );

    let (mut child, _, _) = parent
        .spawn_named_child(&[("counter".into(), offer)], 1 << 16)
        .expect("spawn");
    let ch = child.resolve_cap_name("counter").expect("named");
    let ctid = child.resolve_guest_impl(ch).unwrap().type_id;
    assert_eq!(
        child.cap_dispatch_slots(ctid, 0, ch, &[], None),
        Ok(vec![2]),
        "the child drives the same instance the parent bumped"
    );
    assert_eq!(
        parent.cap_dispatch_slots(ptid, 0, offer, &[], None),
        Ok(vec![3]),
        "and the parent sees the child's bump"
    );
}

#[test]
fn a_wrap_holds_and_forwards_a_real_capability() {
    // §3.2 v2 wrap: the wirer re-grants its own stdout INTO the provider; the provider's op
    // resolves it by name (from its own data segment) and writes a payload from its OWN
    // window through it — interposition holding real forwarded authority, entirely inside
    // the provider's domain.
    let provider = svm_text::parse_module(
        "memory 16\n\
         data 0 \"hi\"\n\
         data 8 \"out\"\n\
         func () -> (i64) {\n\
         block0():\n\
           vp = i64.const 8\n\
           vn = i64.const 3\n\
           vh = cap.self.resolve vp vn\n\
           vbuf = i64.const 0\n\
           vlen = i64.const 2\n\
           vw = cap.call 0 1 (i64, i64) -> (i64) vh (vbuf, vlen)\n\
           return vw\n\
         }\n",
    )
    .expect("provider parses");
    svm_verify::verify_module(&provider).expect("verifies");

    let mut h = Host::new();
    let out = h.grant_stream(svm_interp::StreamRole::Out);
    let offer = h.wire_impl_instance(&provider, &[0]).expect("offer");
    h.grant_impl_cap(offer, out, "out").expect("grantable");
    let tid = h.resolve_guest_impl(offer).unwrap().type_id;
    assert_eq!(
        h.cap_dispatch_slots(tid, 0, offer, &[], None),
        Ok(vec![2]),
        "the provider's write through the forwarded stream reports 2 bytes"
    );
    // The re-grant shared the wirer's stdout sink, so the provider's write lands in the
    // wirer's captured output.
    assert_eq!(h.stdout_bytes(), b"hi", "payload crossed the wrap");
}

#[test]
fn provider_pays_from_a_drainable_reserve() {
    // §5.3 (resolved): the provider funds its own dispatch compute. The wirer prices the
    // service; a dry reserve makes further calls an inert CapFault the caller can probe,
    // and the wirer reads the meter.
    let provider = counter_provider();
    let mut h = Host::new();
    let offer = h.wire_impl_instance(&provider, &[0]).expect("offer");
    let tid = h.resolve_guest_impl(offer).unwrap().type_id;

    let full = h
        .impl_fuel_remaining(offer)
        .expect("instanced offers meter");
    assert_eq!(h.cap_dispatch_slots(tid, 0, offer, &[], None), Ok(vec![1]));
    let after = h.impl_fuel_remaining(offer).unwrap();
    assert!(after < full, "the call drained what it used");

    // Clamp the reserve below one call's cost: the next dispatch fails closed and drains
    // the remainder; the one after that is refused outright on the empty reserve.
    h.set_impl_fuel_reserve(offer, 3).expect("wirer prices it");
    assert!(h.cap_dispatch_slots(tid, 0, offer, &[], None).is_err());
    assert!(h.cap_dispatch_slots(tid, 0, offer, &[], None).is_err());
    // Top-up restores service — and the counter state survived the dry spell.
    h.set_impl_fuel_reserve(offer, 1 << 20).expect("top-up");
    assert_eq!(h.cap_dispatch_slots(tid, 0, offer, &[], None), Ok(vec![2]));

    // Pure offers have no reserve to meter.
    let pure = h.wire_impl(&offer_funcs(), &[0]).expect("pure");
    assert!(h.impl_fuel_remaining(pure).is_none());
}

#[test]
fn grant_impl_cap_refuses_offers_and_pure_offers() {
    // Acyclicity: a provider can never hold an offer (the deadlock-freedom invariant), and a
    // v1 pure offer has no provider to grant into.
    let provider = counter_provider();
    let mut h = Host::new();
    let instanced = h.wire_impl_instance(&provider, &[0]).expect("instanced");
    let pure = h.wire_impl(&offer_funcs(), &[0]).expect("pure");
    let clock = h.grant_clock();
    assert!(
        h.grant_impl_cap(instanced, pure, "svc").is_none(),
        "offers never nest in providers"
    );
    assert!(
        h.grant_impl_cap(pure, clock, "clk").is_none(),
        "a pure offer has no provider instance"
    );
    assert!(
        h.grant_impl_cap(instanced, clock, "clk").is_some(),
        "a platform cap re-grants fine"
    );
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

// ---- §3.5: grouped imports, coverage binding, reflection, export.handle ----

/// A provider module with a named two-op interface and an offer:
/// op `add(a, b) = a + b` (func 0), op `dbl(a) = a + a` (func 1).
fn provider_module() -> svm_ir::Module {
    svm_text::parse_module(
        "type 0 func (i64, i64) -> (i64)\n\
         type 1 func (i64) -> (i64)\n\
         type 2 interface { add: 0, dbl: 1 }\n\
         export 0 interface \"svc\" 2 { add: 0, dbl: 1 }\n\n\
         func (i64, i64) -> (i64) {\n\
         block0(va: i64, vb: i64):\n\
           vs = i64.add va vb\n\
           return vs\n\
         }\n\n\
         func (i64) -> (i64) {\n\
         block0(va: i64):\n\
           vs = i64.add va va\n\
           return vs\n\
         }\n",
    )
    .expect("provider parses")
}

#[test]
fn grouped_import_coverage_binds_a_subset_with_a_remap() {
    // The parent reifies its own offer (`export.handle` semantics, host-side), grants it to
    // a child under a name; the child's grouped import requires ONLY `dbl` — a subset of the
    // provider's two ops — and coverage binding freezes the remap [0 -> 1].
    let pm = Arc::new(provider_module());
    svm_verify::verify_module(&pm).expect("provider verifies");
    let mut parent = Host::new();
    parent.set_self_module(&pm);
    let h = parent.reify_export(0).expect("reify own offer");
    let (mut child, _ci, _ca) = parent
        .spawn_named_child(&[("svc".into(), h)], 1 << 16)
        .expect("spawn");

    let cm = svm_text::parse_module(
        "type 0 func (i64) -> (i64)\n\
         type 1 interface { dbl: 0 }\n\
         import 0 interface \"svc\" 1\n\n\
         func (i64) -> (i64) {\n\
         block0(va: i64):\n\
           vh = i32.const 0\n\
           vr = call.import 0.dbl vh (va)\n\
           return vr\n\
         }\n",
    )
    .expect("consumer parses");
    svm_verify::verify_module(&cm).expect("consumer verifies");
    child
        .bind_child_manifest(&cm.imports, &cm.types)
        .expect("subset requirement covers");

    // Consumer-local op 0 remaps to provider op 1 (`dbl`) — on both engines.
    let args = [Value::I64(21)];
    let mut fuel = 1_000_000u64;
    let tree = svm_interp::run_with_host(&cm, 0, &args, &mut fuel, &mut child);
    assert_eq!(tree, Ok(vec![Value::I64(42)]), "tree-walker remap dispatch");
    let mut fuel_b = 1_000_000u64;
    let byte =
        svm_interp::bytecode::compile_and_run_with_host(&cm, 0, &args, &mut fuel_b, &mut child)
            .expect("bytecode-eligible");
    assert_eq!(byte, Ok(vec![Value::I64(42)]), "bytecode remap dispatch");
}

#[test]
fn reflection_and_dyn_dispatch_run_over_the_registered_self_module() {
    // One module exercises the §3.5 self surface end to end: `cap.self.type_id` interns its
    // own declared shape, `export.handle` reifies its own offer (memoized), `cap.self.covers`
    // confirms the reified capability covers the shape, and `call.import.dyn` drives it by
    // type-section reference — identically on both engines.
    let src = "type 0 func (i64, i64) -> (i64)\n\
         type 1 func (i64) -> (i64)\n\
         type 2 interface { add: 0, dbl: 1 }\n\
         export 0 interface \"svc\" 2 { add: 0, dbl: 1 }\n\n\
         func (i64, i64) -> (i64) {\n\
         block0(va: i64, vb: i64):\n\
           vs = i64.add va vb\n\
           return vs\n\
         }\n\n\
         func (i64) -> (i64) {\n\
         block0(va: i64):\n\
           vs = i64.add va va\n\
           return vs\n\
         }\n\n\
         func (i64, i64) -> (i64) {\n\
         block0(va: i64, vb: i64):\n\
           vt = cap.self.type_id 2\n\
           vh = export.handle 0\n\
           vc = cap.self.covers vh 2\n\
           vr = call.import.dyn 2.add vh (va, vb)\n\
           vcov = i64.extend_i32_u vc\n\
           vsum = i64.add vr vcov\n\
           return vsum\n\
         }\n";
    let m = Arc::new(svm_text::parse_module(src).expect("parses"));
    svm_verify::verify_module(&m).expect("verifies");
    // Dynamic-mode dispatch costs the manifest-completeness bit, like cap.call.
    assert!(!svm_verify::manifest_complete(&m), "dyn mode costs the bit");

    let run = |use_bytecode: bool| {
        let mut h = Host::new();
        h.set_self_module(&m);
        let args = [Value::I64(40), Value::I64(1)];
        let mut fuel = 1_000_000u64;
        if use_bytecode {
            svm_interp::bytecode::compile_and_run_with_host(&m, 2, &args, &mut fuel, &mut h)
                .expect("bytecode-eligible")
        } else {
            svm_interp::run_with_host(&m, 2, &args, &mut fuel, &mut h)
        }
    };
    // add(40, 1) = 41, covers = 1 → 42.
    assert_eq!(run(false), Ok(vec![Value::I64(42)]), "tree-walker");
    assert_eq!(run(true), Ok(vec![Value::I64(42)]), "bytecode engine");

    // Re-reifying returns the same handle (one shared service state per domain).
    let mut h = Host::new();
    h.set_self_module(&m);
    let h1 = h.reify_export(0).expect("first reify");
    let h2 = h.reify_export(0).expect("second reify");
    assert_eq!(h1, h2, "export.handle is memoized per offer");
    // An unregistered domain fails closed (probeable, never a hang).
    let mut bare = Host::new();
    assert!(bare.reify_export(0).is_err(), "no self module ⇒ CapFault");
}

#[test]
fn grouped_attach_runs_the_coverage_walk_and_refreshes_the_remap() {
    // A grouped **rebindable** slot starts empty; the guest later discovers a capability and
    // attaches it. §3.5: attach is the coverage walk against the slot's retained manifest
    // requirement — a covering offer binds (remap refreshed), a non-covering one is -EINVAL.
    let pm = Arc::new(provider_module());
    let mut parent = Host::new();
    parent.set_self_module(&pm);
    let h = parent.reify_export(0).expect("reify");
    let (mut child, _ci, _ca) = parent
        .spawn_named_child(&[("later".into(), h)], 1 << 16)
        .expect("spawn");

    // The consumer requires only `dbl`, rebindable, under a name nothing matches at spawn.
    let cm = svm_text::parse_module(
        "type 0 func (i64) -> (i64)\n\
         type 1 interface { dbl: 0 }\n\
         import 0 interface \"nothing\" 1 rebindable\n\n\
         func (i64) -> (i64) {\n\
         block0(va: i64):\n\
           vh = i32.const 0\n\
           vr = call.import 0.dbl vh (va)\n\
           return vr\n\
         }\n",
    )
    .expect("consumer parses");
    child
        .bind_child_manifest(&cm.imports, &cm.types)
        .expect("rebindable withhold = empty slot");
    assert!(child.import_binding(0).is_none(), "starts empty");

    // A non-covering attach (a plain module grant is no guest impl) fails closed, probeable.
    let bogus = child.grant_module(&cm);
    let r = child
        .cap_dispatch_slots(
            svm_ir::CAP_IMPORT_ATTACH_TYPE_ID,
            0,
            0,
            &[bogus as i64],
            None,
        )
        .expect("attach dispatch itself succeeds");
    assert!(r[0] < 0, "non-covering handle refuses with -errno");
    assert!(child.import_binding(0).is_none(), "slot still empty");

    // The discovered offer covers the subset requirement: attach binds + remaps [0 -> 1].
    let disc = child.resolve_cap_name("later").expect("discovered");
    let r = child
        .cap_dispatch_slots(
            svm_ir::CAP_IMPORT_ATTACH_TYPE_ID,
            0,
            0,
            &[disc as i64],
            None,
        )
        .expect("attach dispatch");
    assert_eq!(r, vec![0], "covering attach succeeds");
    let args = [Value::I64(21)];
    let mut fuel = 1_000_000u64;
    let tree = svm_interp::run_with_host(&cm, 0, &args, &mut fuel, &mut child);
    assert_eq!(tree, Ok(vec![Value::I64(42)]), "attached remap dispatch");
}
