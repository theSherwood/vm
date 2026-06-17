//! Host-assisted dynamic linking, end to end over a **serialized** unit (DYNLINK.md capstone, the
//! host-assisted resolve path). The companion `dynlink_runtime.rs` resolves an *in-memory* `Module`
//! with `resolve_imports_with` inside the test harness; here the plugin is **serialized to bytes
//! while its symbol is still unresolved** (a `.so` with an undefined reference), and the *host's*
//! compile path — [`svm_run::jit_resolve_and_validate`] — decodes it, binds the import by name
//! against a guest-controlled symbol table, re-verifies, and only then compiles. This is what
//! `vm_dlopen` will call: the loader ships an IR blob + a symbol table, the host does the rewrite.
//!
//! Two new pieces are exercised: the v2 binary codec carries the §7 import section + `call.import`
//! (so a unit can be serialized *unresolved*), and the resolve runs **before** verification, so a
//! mis-link is caught by re-verification rather than trusted ("rewrite-then-verify").

use svm_encode::{decode_module, encode_module};
use svm_ir::{Inst, Resolved, ResolvedCap, DEFAULT_RESERVED_LOG2};
use svm_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};
use svm_run::{encode_symbol_table, jit_blob_validator, jit_resolve_and_validate};
use svm_text::parse_module;
use svm_verify::verify_module;

fn compile_host(src: &str) -> CompiledModule {
    let m = parse_module(src).expect("parse host");
    verify_module(&m).expect("verify host");
    CompiledModule::compile(
        &m,
        0,
        INERT_CAP_THUNK,
        core::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("compile host program")
}

/// A plugin authored against host `F` purely **by name** — its import handle is the `ConstI32`
/// placeholder a slot binding patches into the `call_indirect` index.
const PLUGIN_SRC: &str = "func (i32, i32) -> (i32) {\n\
     block0(v0: i32, v1: i32):\n\
     \x20 v2 = i32.const 0\n\
     \x20 v3 = call.import \"F\" (i32, i32) -> (i32) v2 (v0, v1)\n\
     \x20 return v3\n\
     }\n";

/// The §7 import section + `call.import` survive the binary form (v2 codec): a serialized unit can
/// carry an **unresolved** symbol, the precondition for shipping a `.so`-shaped blob to a loader.
#[test]
fn import_section_round_trips_through_the_binary_codec() {
    let plugin = parse_module(PLUGIN_SRC).expect("parse plugin");
    assert_eq!(plugin.imports.len(), 1, "the plugin imports F by name");
    let back = decode_module(&encode_module(&plugin)).expect("decode");
    assert_eq!(
        back, plugin,
        "the import section + call.import must round-trip through the binary form"
    );
    assert_eq!(back.imports[0].name, "F");
}

#[test]
fn host_assisted_resolve_links_a_serialized_plugin_by_name() {
    // The host program: `F(a,b) = a*2 + b`, at function/table slot 0 (also the entry).
    let mut cm = compile_host(
        "func (i32, i32) -> (i32) {\n\
         block0(v0: i32, v1: i32):\n\
         \x20 v2 = i32.const 2\n\
         \x20 v3 = i32.mul v0 v2\n\
         \x20 v4 = i32.add v3 v1\n\
         \x20 return v4\n\
         }\n",
    );

    // Serialize the plugin **with its import unresolved** — exactly the bytes a guest `vm_dlopen`
    // would hand the host.
    let blob = encode_module(&parse_module(PLUGIN_SRC).expect("parse plugin"));

    // The host's resolving compile path: decode → bind "F" → host slot 0 (the guest's symbol table)
    // → re-verify → funcs. No `resolve_imports_with` in the harness; the host did the rewrite.
    let funcs = jit_resolve_and_validate(&blob, None, |n| (n == "F").then_some(Resolved::Slot(0)))
        .expect("host resolves the plugin's import by name");

    // Compile the resolved unit at runtime against the host's live table, then call it: it dispatches
    // through the shared table to the host's F — F(10,3) = 23.
    let ptrs = cm
        .define_extra(&funcs)
        .expect("define_extra (compile the plugin)");
    let (out, _) =
        unsafe { cm.run_extra(ptrs[0].tramp, 2, 1, &[10, 3], None) }.expect("run plugin");
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[23]),
        "the host-resolved plugin reached F through the table: expected 23, got {out:?}"
    );
}

/// Fail-closed: a symbol the table doesn't carry is unresolvable, so the resolve fails before
/// verify/compile — nothing is installed (`-EINVAL`), never a silent miscompile.
#[test]
fn unresolved_symbol_fails_closed() {
    let blob = encode_module(&parse_module(PLUGIN_SRC).expect("parse plugin"));
    // The guest's symbol table has no "F".
    let r = jit_resolve_and_validate(&blob, None, |_| None);
    assert_eq!(r.err(), Some(-22), "an unresolved import must fail closed");
}

/// Fail-closed: a garbled blob is rejected by `decode_module` (the untrusted-input TCB) before any
/// resolution, so a forged byte string can never reach the verifier or a backend.
#[test]
fn malformed_blob_fails_closed() {
    let r = jit_resolve_and_validate(&[0xde, 0xad, 0xbe, 0xef], None, |_| Some(Resolved::Slot(0)));
    assert_eq!(r.err(), Some(-22), "a malformed blob must fail closed");
}

/// A symbol can resolve to a **host capability**, not only a table slot: the symbol table's `Cap`
/// kind (the `1` byte) binds a named import to a `(type_id, op)` capability, which the loader lowers
/// to a `cap.call` — so a loaded unit (a plugin) can reach a *host service* by name (e.g. `"write"`),
/// not just another loaded unit. This drives the symbol table's `Cap` branch through the real
/// `jit_blob_validator` byte path (the same gate the `compile_linked` op uses), proving the wire
/// form + lowering are exercised end to end. The unit takes the capability handle as `arg0` (the
/// import's handle operand, kept across the rewrite — unlike a `Slot`, a `Cap` needs no placeholder).
#[test]
fn symbol_table_resolves_a_capability_import_by_name() {
    let unit = "func (i32, i64, i64) -> (i64) {\n\
                block0(v0: i32, v1: i64, v2: i64):\n\
                \x20 v3 = call.import \"write\" (i64, i64) -> (i64) v0 (v1, v2)\n\
                \x20 return v3\n\
                }\n";
    let blob = encode_module(&parse_module(unit).expect("parse cap-importing unit"));

    // The loader binds "write" to a host capability (type_id 3, op 1) via the symbol table's Cap kind.
    let symtab =
        encode_symbol_table(&[("write", Resolved::Cap(ResolvedCap { type_id: 3, op: 1 }))]);
    let funcs = jit_blob_validator(&blob, None, &symtab)
        .expect("the capability import resolves by name and re-verifies");

    // Resolution lowered `call.import "write"` to a concrete `cap.call 3 1` (no import survives).
    let lowered_to_cap_call = funcs[0].blocks.iter().flat_map(|b| &b.insts).any(|i| {
        matches!(
            i,
            Inst::CapCall {
                type_id: 3,
                op: 1,
                ..
            }
        )
    });
    assert!(
        lowered_to_cap_call,
        "the named capability import must lower to `cap.call 3 1`"
    );
}
