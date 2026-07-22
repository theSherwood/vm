//! Guest-driven **host-assisted dynamic linking** (DESIGN.md §22): the `Jit` capability's new
//! `compile_linked` op (op 5). A guest submits a serialized unit that still carries **unresolved §7
//! imports** plus a **symbol-table buffer** (`name → slot`); the host resolves the imports by name
//! against that table, re-verifies, and compiles — all in-sandbox, driven by guest code. This is the
//! cap-op the guest-side `vm_dlopen` will call; the harness-level `dynlink_repl.rs` is its spec.
//!
//! Every test is **differential**: the exact same guest runs on the reference interpreter and the
//! JIT (with a byte-identical powerbox), and the outcomes + final memory must agree — the op behaves
//! identically on both backends, like every other `Jit` op.

use svm_encode::encode_module;
use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_ir::{Resolved, DEFAULT_RESERVED_LOG2};
use svm_jit::{JitOutcome, TrapKind};
use svm_run::{encode_symbol_table, grant_jit, jit_cap_run};
use svm_text::parse_module;
use svm_verify::verify_module;

/// A self-contained blob (no imports): parse, verify, encode.
fn blob(src: &str) -> Vec<u8> {
    let m = parse_module(src).expect("parse blob");
    verify_module(&m).expect("verify blob");
    encode_module(&m)
}

/// A unit serialized **with its §7 imports still unresolved** — `verify` would reject it (imports
/// are resolved before verify), so we only parse + encode. This is the `.so` a loader resolves.
fn unresolved_blob(src: &str) -> Vec<u8> {
    encode_module(&parse_module(src).expect("parse unresolved blob"))
}

/// Run `guest_src`'s func 0 on both backends over the prepared `init` window image (handle = arg 0,
/// then `user_args`), reserving a `2^table_log2`-slot `call_indirect` table identically on each;
/// assert the outcomes + final memory agree and return the JIT's `(outcome, final_mem)`.
fn diff(guest_src: &str, init: &[u8], user_args: &[i64], table_log2: u8) -> (JitOutcome, Vec<u8>) {
    let m = parse_module(guest_src).expect("parse guest");
    verify_module(&m).expect("verify guest");

    // Interpreter.
    let mut host_i = Host::new();
    let h_i = grant_jit(&mut host_i, &m, table_log2);
    let mut iargs = vec![Value::I32(h_i)];
    iargs.extend(user_args.iter().map(|&a| Value::I32(a as i32)));
    let mut fuel = 50_000_000u64;
    let (ires, imem) = run_capture_reserved_with_host(
        &m,
        0,
        &iargs,
        &mut fuel,
        init,
        DEFAULT_RESERVED_LOG2,
        &mut host_i,
    );

    // JIT — a fresh Host configured identically (deterministic grant ⇒ identical handle value).
    let mut host_j = Host::new();
    let h_j = grant_jit(&mut host_j, &m, table_log2);
    assert_eq!(h_i, h_j, "identical powerbox setup mints identical handles");
    let mut jargs = vec![h_j as i64];
    jargs.extend_from_slice(user_args);
    let (jout, jmem) = jit_cap_run(
        &m,
        0,
        &jargs,
        init,
        DEFAULT_RESERVED_LOG2,
        table_log2,
        &mut host_j,
    )
    .expect("jit run");

    match (&ires, &jout) {
        (Ok(vals), JitOutcome::Returned(slots)) => {
            assert_eq!(vals.len(), slots.len(), "result arity");
            for (v, s) in vals.iter().zip(slots) {
                let iv = match v {
                    Value::I32(x) => *x as i64,
                    Value::I64(x) => *x,
                    other => panic!("scalar result expected, got {other:?}"),
                };
                assert_eq!(iv, *s, "interp {ires:?} != jit {jout:?}");
            }
        }
        (Err(Trap::CapFault), JitOutcome::Trapped(TrapKind::CapFault))
        | (Err(Trap::IndirectCallType), JitOutcome::Trapped(TrapKind::IndirectCallType)) => {}
        other => panic!("backends disagree: {other:?}"),
    }
    assert_eq!(imem, jmem, "final memory must be byte-identical");
    (jout, jmem)
}

// Window layout shared by the guests below.
const SVC_OFF: usize = 4096;
const UNIT_OFF: usize = 6144;
const SYMTAB_OFF: usize = 8192;

/// `service(a, b) = a*a + b` — a self-contained unit the guest installs into the table.
const SERVICE: &str = "memory 16\nfunc (i32, i32) -> (i32) {\n\
    block 0 (v0: i32, v1: i32) {\n  v2 = i32.mul v0 v0\n  v3 = i32.add v2 v1\n  return v3\n  }\n}\n";

/// `unit(a, b) = F(a, b) + 100`, where `F` is an **unresolved import** the loader binds by name.
const UNIT: &str = "memory 16\nfunc (i32, i32) -> (i32) {\n\
    block 0 (v0: i32, v1: i32) {\n  v2 = i32.const 0\n\
    \x20 v3 = call.sym \"F\" (i32, i32) -> (i32) v2 (v0, v1)\n\
    \x20 v4 = i32.const 100\n  v5 = i32.add v3 v4\n  return v5\n  }\n}\n";

/// Seed an init image with the service at [`SVC_OFF`] and the unit at [`UNIT_OFF`].
fn seed_service_and_unit(svc: &[u8], unit: &[u8]) -> Vec<u8> {
    let mut init = vec![0u8; UNIT_OFF + unit.len()];
    init[SVC_OFF..SVC_OFF + svc.len()].copy_from_slice(svc);
    init[UNIT_OFF..UNIT_OFF + unit.len()].copy_from_slice(unit);
    init
}

/// The guest for the full install→link→invoke flow (shared by the success case and the type-mismatch
/// case): compile the service at [`SVC_OFF`], install it, build the symbol table binding `"F"` to the
/// **install slot** at [`SYMTAB_OFF`] (`[count=1, namelen=1, 'F', kind=0/Slot, slot]`; `'F'`=70, a
/// slot < 128 is one uleb byte), `compile_linked` the unit at [`UNIT_OFF`] against it, and invoke
/// `unit(a, b)`. The slot is read back from install (the loader's real pattern — never hard-coded).
fn install_link_invoke_guest(svc_len: usize, unit_len: usize) -> String {
    format!(
        "memory 16\nfunc (i32, i32, i32) -> (i32) {{\nblock 0 (v0: i32, v1: i32, v2: i32) {{\n\
         \x20 v3 = i64.const {svc_off}\n  v4 = i64.const {svc_len}\n\
         \x20 v5 = cap.call 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n\
         \x20 v6 = cap.call 11 3 (i64) -> (i64) v0 (v5)\n\
         \x20 v7 = i64.const {st}\n  v8 = i32.const 1\n  i32.store8 v7 v8\n\
         \x20 v9 = i64.const {st1}\n  i32.store8 v9 v8\n\
         \x20 v10 = i64.const {st2}\n  v11 = i32.const 70\n  i32.store8 v10 v11\n\
         \x20 v12 = i64.const {st3}\n  v13 = i32.const 0\n  i32.store8 v12 v13\n\
         \x20 v14 = i64.const {st4}\n  v15 = i32.wrap_i64 v6\n  i32.store8 v14 v15\n\
         \x20 v16 = i64.const {unit_off}\n  v17 = i64.const {unit_len}\n\
         \x20 v18 = i64.const {st}\n  v19 = i64.const 5\n\
         \x20 v20 = cap.call 11 5 (i64, i64, i64, i64) -> (i64) v0 (v16, v17, v18, v19)\n\
         \x20 v21 = cap.call 11 1 (i64, i32, i32) -> (i32) v0 (v20, v1, v2)\n\
         \x20 return v21\n  }}\n}}\n",
        svc_off = SVC_OFF,
        unit_off = UNIT_OFF,
        st = SYMTAB_OFF,
        st1 = SYMTAB_OFF + 1,
        st2 = SYMTAB_OFF + 2,
        st3 = SYMTAB_OFF + 3,
        st4 = SYMTAB_OFF + 4,
    )
}

/// The flagship: the **full guest-driven REPL flow**, end to end, on both backends. The guest
/// compiles `service`, installs it (getting a table slot), **builds a symbol table in its own
/// window** binding `"F"` to that slot, `compile_linked`s the unit — which imports `F` by name —
/// against it, and finally invokes the unit. The unit reaches the installed service purely by name:
/// `unit(5, 2) = service(5, 2) + 100 = (25 + 2) + 100 = 127`. Nothing is hand-resolved in the
/// harness; the *guest* delivers the symbol table and the *host* does the rewrite-then-verify.
#[test]
fn guest_compiles_links_and_invokes_by_name_across_backends() {
    let svc = blob(SERVICE);
    let unit = unresolved_blob(UNIT);
    let init = seed_service_and_unit(&svc, &unit);
    let guest = install_link_invoke_guest(svc.len(), unit.len());

    let (out, _) = diff(&guest, &init, &[5, 2], 4);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[127]),
        "the guest-linked unit reached the installed service by name: expected 127, got {out:?}"
    );
}

/// The security edge: linking a symbol to a slot holding a **wrong-typed** function does not produce
/// a type-confused call — it **traps** at the call site. Here the same guest installs a *one*-arg
/// service `(i32)->(i32)` and binds `"F"` (which the unit imports as `(i32,i32)->(i32)`) to it. The
/// link succeeds (resolution only rewrites the import to `call_indirect <slot>` and re-verifies — a
/// `call_indirect` is well-typed IR), but at invoke the slot's `type_id` doesn't match the call's, so
/// the masked dispatch faults `IndirectCallType`, identically on both backends. The loader cannot be
/// tricked into an out-of-type dispatch — the §3c table check carries the safety, exactly as for any
/// slot the guest already controls.
#[test]
fn linking_to_a_wrong_typed_slot_traps_not_confuses() {
    // A service of the WRONG arity for the import: (i32) -> (i32).
    let svc = blob("memory 16\nfunc (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n");
    let unit = unresolved_blob(UNIT); // imports F as (i32, i32) -> (i32)
    let init = seed_service_and_unit(&svc, &unit);
    let guest = install_link_invoke_guest(svc.len(), unit.len());

    let (out, _) = diff(&guest, &init, &[5, 2], 4);
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::IndirectCallType)),
        "a mis-typed link must trap (IndirectCallType), never dispatch type-confused: got {out:?}"
    );
}

/// A guest that only `compile_linked`s the unit (at [`UNIT_OFF`]) against the symbol table at
/// [`SYMTAB_OFF`] and returns the raw result (handle or `-errno`) — for the fail-closed cases.
fn compile_linked_only(unit_len: usize, symtab_len: usize) -> String {
    format!(
        "memory 16\nfunc (i32) -> (i64) {{\nblock 0 (v0: i32) {{\n\
         \x20 v1 = i64.const {unit_off}\n  v2 = i64.const {unit_len}\n\
         \x20 v3 = i64.const {st}\n  v4 = i64.const {symtab_len}\n\
         \x20 v5 = cap.call 11 5 (i64, i64, i64, i64) -> (i64) v0 (v1, v2, v3, v4)\n\
         \x20 return v5\n  }}\n}}\n",
        unit_off = UNIT_OFF,
        st = SYMTAB_OFF,
    )
}

/// Seed an init image with the unit at [`UNIT_OFF`] and a raw symbol-table byte string at
/// [`SYMTAB_OFF`].
fn seed_unit_and_symtab(unit: &[u8], symtab: &[u8]) -> Vec<u8> {
    let end = (UNIT_OFF + unit.len()).max(SYMTAB_OFF + symtab.len());
    let mut init = vec![0u8; end];
    init[UNIT_OFF..UNIT_OFF + unit.len()].copy_from_slice(unit);
    init[SYMTAB_OFF..SYMTAB_OFF + symtab.len()].copy_from_slice(symtab);
    init
}

/// Fail-closed: the unit imports `F`, but the symbol table is **empty** (`count = 0`) — `F` is
/// unresolvable, so the resolve fails before verify/compile: `-EINVAL`, identically on both backends.
#[test]
fn compile_linked_unresolved_symbol_fails_closed() {
    let unit = unresolved_blob(UNIT);
    let symtab = encode_symbol_table(&[]); // a single `0` count byte
    let init = seed_unit_and_symtab(&unit, &symtab);
    let guest = compile_linked_only(unit.len(), symtab.len());
    let (out, _) = diff(&guest, &init, &[], 0);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[-22]),
        "an unresolved import must fail closed (-EINVAL) on both backends, got {out:?}"
    );
}

/// Fail-closed: a **malformed** symbol table (a bad `kind` byte) is rejected by the host decoder
/// before any IR is touched: `-EINVAL`, identically on both backends.
#[test]
fn compile_linked_malformed_symtab_fails_closed() {
    let unit = unresolved_blob(UNIT);
    // count=1, namelen=1, 'F', kind=9 (unknown) → the decoder rejects it.
    let symtab = vec![1u8, 1, 70, 9];
    let init = seed_unit_and_symtab(&unit, &symtab);
    let guest = compile_linked_only(unit.len(), symtab.len());
    let (out, _) = diff(&guest, &init, &[], 0);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[-22]),
        "a malformed symbol table must fail closed (-EINVAL) on both backends, got {out:?}"
    );
}

/// A static symbol table (built host-side) resolves the same way as the guest-built one: binding
/// `F → Slot(1)` lets `compile_linked` succeed (a real handle, ≥ 0) even before the slot is filled —
/// resolution bakes the slot into a `call_indirect`; a still-empty slot only traps at *invoke* time.
#[test]
fn compile_linked_with_a_resolvable_table_returns_a_handle() {
    let unit = unresolved_blob(UNIT);
    let symtab = encode_symbol_table(&[("F", Resolved::Slot(1))]);
    let init = seed_unit_and_symtab(&unit, &symtab);
    let guest = compile_linked_only(unit.len(), symtab.len());
    // Reserve a table (log2=4) so Slot(1) is a valid index the verifier/compile accept.
    let (out, _) = diff(&guest, &init, &[], 4);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s[0] >= 0),
        "a resolvable import compiles to a handle on both backends, got {out:?}"
    );
}
